# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Serve a browser-based viewer for a directory of Isaac Lab-style recorder HDF5 files.

Lets you browse the group/dataset tree, play back RGB camera observations frame-by-frame, and
plot numeric datasets (actions, joint targets, raw teleop signals) as line charts -- without
loading whole multi-gigabyte files into memory, since every value is read on demand via ``h5py``.

The script has zero simulation dependency and only requires ``h5py`` and ``numpy``.

Example
-------
.. code-block:: bash

    python isaaclab_arena/scripts/tools/hdf5_viewer/server.py --dir /path/to/hdf5_dir
    # then open http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import numpy as np
import struct
import threading
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import h5py

_STATIC_DIR = Path(__file__).parent / "static"
_MAX_SERIES_POINTS = 20_000
_FILE_CACHE_SIZE = 3
# This dataset's camera chunks are shaped like (825, 14, 14, 1): each chunk spans 825 frames but
# only a 14x14 pixel patch of one channel, so a single 224x224x3 frame is scattered across
# 16*16*3 = 768 gzip chunks. HDF5's default per-dataset chunk cache is only 1 MiB, far too small
# to hold those 768 chunks (~120 MiB decompressed), so without a bigger cache every single frame
# read decompresses all 768 chunks from scratch (~220ms). With a cache sized to fit one full
# "layer" of chunks, only the first frame of each 825-frame block pays that cost; the other 824
# frames in the block hit the cache (~2ms). See also gzip -> raw byte layout notes in _encode_png.
_CHUNK_CACHE_BYTES = 256 * 1024 * 1024
_CHUNK_CACHE_SLOTS = 100_003  # HDF5 docs recommend ~100x the number of chunks expected in cache


class _Hdf5FileCache:
    """Keeps a handful of HDF5 files open (with an enlarged chunk cache) across requests.

    Playback re-requests one frame per tick from the same file; re-opening large (multi-GB,
    many-chunk) files on every request dominates request latency far more than reading the frame
    itself, so we keep the last few opened files around instead of a ``with h5py.File(...)`` per
    request. h5py serializes access to the underlying HDF5 C library with a global lock, so
    sharing one handle across the threading server's worker threads is safe, just not parallel.
    """

    def __init__(self, max_open: int):
        self._max_open = max_open
        self._lock = threading.Lock()
        self._files: dict[Path, h5py.File] = {}
        # Re-opening a dataset (``file[dataset_path]``) creates a new low-level HDF5 dataset
        # handle with its own *empty* chunk cache -- keeping the file open is not enough to get
        # the warm-cache benefit above, we have to reuse the actual h5py.Dataset object too.
        # Entries are dropped whenever their owning file is evicted/closed.
        self._datasets: dict[tuple[Path, str], h5py.Dataset] = {}

    def _get_file_locked(self, path: Path) -> h5py.File:
        f = self._files.pop(path, None)
        if f is None:
            if len(self._files) >= self._max_open:
                oldest_path = next(iter(self._files))  # dict preserves insertion order
                self._files.pop(oldest_path).close()
                for key in [k for k in self._datasets if k[0] == oldest_path]:
                    del self._datasets[key]
            f = h5py.File(path, "r", rdcc_nbytes=_CHUNK_CACHE_BYTES, rdcc_nslots=_CHUNK_CACHE_SLOTS)
        self._files[path] = f  # re-insert so it's the most-recently-used entry
        return f

    def get_file(self, path: Path) -> h5py.File:
        with self._lock:
            return self._get_file_locked(path)

    def get_dataset(self, path: Path, dataset_path: str) -> h5py.Dataset:
        with self._lock:
            f = self._get_file_locked(path)
            key = (path, dataset_path)
            dataset = self._datasets.get(key)
            if dataset is None:
                if dataset_path not in f:
                    raise _ApiError(f"path not found in file: {dataset_path!r}", status=404)
                obj = f[dataset_path]
                if not isinstance(obj, h5py.Dataset):
                    raise _ApiError(f"not a dataset: {dataset_path!r}")
                dataset = obj
                self._datasets[key] = dataset
            return dataset

    def close_all(self):
        with self._lock:
            for f in self._files.values():
                f.close()
            self._files.clear()
            self._datasets.clear()


_file_cache = _Hdf5FileCache(_FILE_CACHE_SIZE)


def _to_jsonable(value):
    """Recursively convert numpy/h5py attribute values into plain JSON-serializable types."""
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _node_attrs(obj: h5py.Group | h5py.Dataset) -> dict:
    return {key: _to_jsonable(obj.attrs[key]) for key in obj.attrs.keys()}


def _build_tree(obj: h5py.Group | h5py.Dataset, name: str) -> dict:
    if isinstance(obj, h5py.Dataset):
        return {
            "name": name,
            "type": "dataset",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "attrs": _node_attrs(obj),
        }
    children = [_build_tree(obj[key], key) for key in obj.keys()]
    return {
        "name": name,
        "type": "group",
        "attrs": _node_attrs(obj),
        "children": children,
    }


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    chunk = tag + data
    crc = zlib.crc32(chunk) & 0xFFFFFFFF
    return struct.pack("!I", len(data)) + chunk + struct.pack("!I", crc)


def _encode_png(frame: np.ndarray) -> bytes:
    """Minimal RGB8 PNG encoder so the viewer needs no imaging library beyond numpy/zlib."""
    assert frame.ndim == 3 and frame.shape[2] == 3, f"expected HxWx3 RGB frame, got {frame.shape}"
    height, width, _ = frame.shape
    # Prepend a zero "filter type: None" byte to each scanline via one array op instead of a
    # per-row Python loop -- cheap next to the chunk decompression cost, but adds up during
    # sequential playback otherwise.
    scanlines = np.zeros((height, 1 + width * 3), dtype=np.uint8)
    scanlines[:, 1:] = frame.reshape(height, width * 3)
    # Low compression level: this is served over localhost, so encoding speed (frame-to-frame
    # playback latency) matters far more here than payload size.
    compressed = zlib.compress(scanlines.tobytes(), level=1)
    ihdr = struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit depth, color type 2 = RGB
    return (
        b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", compressed) + _png_chunk(b"IEND", b"")
    )


class _ApiError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def make_handler(data_dir: Path) -> type[BaseHTTPRequestHandler]:
    """Build a request handler bound to ``data_dir`` (kept out of instance state)."""

    def resolve_file(name: str) -> Path:
        if not name or "/" in name or "\\" in name or name in (".", ".."):
            raise _ApiError(f"invalid file name: {name!r}")
        path = data_dir / name
        if not path.is_file():
            raise _ApiError(f"file not found: {name!r}", status=404)
        return path

    class Handler(BaseHTTPRequestHandler):
        server_version = "HDF5Viewer/1.0"

        def log_message(self, fmt, *args):  # quiet default access logging
            pass

        def _send_json(self, payload, status: int = 200):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_bytes(self, body: bytes, content_type: str, status: int = 200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_static(self, rel_path: str):
            path = (_STATIC_DIR / rel_path).resolve()
            if _STATIC_DIR.resolve() not in path.parents and path != _STATIC_DIR.resolve():
                self._send_json({"error": "not found"}, status=404)
                return
            if not path.is_file():
                self._send_json({"error": "not found"}, status=404)
                return
            content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            self._send_bytes(path.read_bytes(), content_type)

        def do_GET(self):  # noqa: N802 (http.server API)
            parsed = urlparse(self.path)
            query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            try:
                if parsed.path in ("/", "/index.html"):
                    self._send_static("index.html")
                elif parsed.path.startswith("/static/"):
                    self._send_static(parsed.path[len("/static/") :])
                elif parsed.path == "/api/files":
                    self._handle_files()
                elif parsed.path == "/api/tree":
                    self._handle_tree(query)
                elif parsed.path == "/api/series":
                    self._handle_series(query)
                elif parsed.path == "/api/frame":
                    self._handle_frame(query)
                else:
                    self._send_json({"error": f"unknown route: {parsed.path}"}, status=404)
            except _ApiError as e:
                self._send_json({"error": str(e)}, status=e.status)
            except Exception as e:  # keep the server alive on unexpected per-request failures
                self._send_json({"error": f"internal error: {e}"}, status=500)

        def _handle_files(self):
            names = sorted(p.name for p in data_dir.glob("*.hdf5"))
            self._send_json({"dir": str(data_dir), "files": names})

        def _handle_tree(self, query: dict):
            file_path = resolve_file(query.get("file", ""))
            f = _file_cache.get_file(file_path)
            tree = _build_tree(f, "/")
            self._send_json(tree)

        def _handle_series(self, query: dict):
            file_path = resolve_file(query.get("file", ""))
            dataset_path = query.get("path", "")
            column = query.get("column")
            dataset = _file_cache.get_dataset(file_path, dataset_path)
            if dataset.dtype.kind not in "fiu" or dataset.ndim > 2:
                raise _ApiError(f"dataset is not a plottable 1D/2D numeric series: {dataset_path!r}")
            array = dataset[()]
            if column is not None:
                array = array[:, int(column)]
            if array.shape[0] > _MAX_SERIES_POINTS:
                raise _ApiError(f"series too long to plot ({array.shape[0]} rows)")
            self._send_json({
                "path": dataset_path,
                "shape": list(dataset.shape),
                "dtype": str(dataset.dtype),
                "data": array.astype(np.float64).tolist(),
            })

        def _handle_frame(self, query: dict):
            file_path = resolve_file(query.get("file", ""))
            dataset_path = query.get("path", "")
            index_str = query.get("index")
            if index_str is None:
                raise _ApiError("missing 'index' query parameter")
            index = int(index_str)
            dataset = _file_cache.get_dataset(file_path, dataset_path)
            if dataset.ndim != 4 or dataset.shape[3] != 3 or dataset.dtype != np.uint8:
                raise _ApiError(f"not an RGB frame stack (T,H,W,3) uint8 dataset: {dataset_path!r}")
            if not (0 <= index < dataset.shape[0]):
                raise _ApiError(f"index {index} out of range [0, {dataset.shape[0]})")
            frame = dataset[index]
            self._send_bytes(_encode_png(frame), "image/png")

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, type=Path, help="Directory containing .hdf5 files to browse.")
    parser.add_argument("--host", default="127.0.0.1", help="Host/interface to bind the server to.")
    parser.add_argument("--port", type=int, default=8765, help="Port to serve the viewer on.")
    args = parser.parse_args()

    data_dir = args.dir.expanduser().resolve()
    assert data_dir.is_dir(), f"{data_dir} is not a directory"

    handler = make_handler(data_dir)
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving HDF5 viewer for {data_dir} at http://{args.host}:{args.port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        _file_cache.close_all()


if __name__ == "__main__":
    main()
