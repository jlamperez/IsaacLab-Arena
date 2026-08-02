# replay_data

Tooling built while diagnosing the `gr00t_dex1_eef_closedloop_policy`/`gr00t_dex1_eef_replay_policy`
pipeline against the BitRobot `G1_WBT_Dex1_Building-Children-Table` dataset. The scripts here are
tracked; the data they produce (`*.npz`, `debug_chunks/`) is gitignored and regenerated on demand.

- **`build_episode_actions_npz.py`** — extracts one recorded episode's `ee_action`/`hand_cmd`/
  `navigate_cmd` from the raw LeRobot parquet into a `.npz` that `gr00t_dex1_eef_replay_policy.py`
  replays open-loop (bypassing GR00T entirely). Needs the Isaac-GR00T venv (pandas/scipy).

- **`patch_modality_json.py`** — adds the `ee_state_left`/`ee_state_right`/`ee_action_left`/
  `ee_action_right` joint groups to the dataset's own `meta/modality.json`. `ee_state`/`ee_action`
  are 12-D dual-arm columns, but `EndEffectorPose` (and our per-arm `ActionConfig`s in
  `g1_dex1_ikea_data_gr00t_n_1_7_config.py`) model one end-effector at a time. Idempotent, plain
  stdlib -- rerun any time the dataset gets re-downloaded fresh.

- **`generate_relative_stats_fast.py`** — computes the `meta/relative_stats.json` normalization
  entries for `ee_action_left`/`ee_action_right`. Vectorized reimplementation of Isaac-GR00T's own
  `gr00t/data/stats.py:generate_rel_stats` (same `T_ref^-1 @ T_action` math, batched with scipy
  instead of one `EndEffectorPose` per frame) -- the original takes ~90-120s/episode
  (~1 day for the full dataset), this takes ~15 min.

- **`test_relative_stats_fast.py`** — checks `generate_relative_stats_fast.py`'s output against
  Isaac-GR00T's own (slow) `RelativeActionLoader` on one real episode before trusting the fast
  version dataset-wide. Run this after touching the vectorized math.

- **`validate_eef_composition.py`** / **`validate_eef_xyz_euler_pipeline.py`** — offline checks that
  `ActionType.EEF` + `ActionFormat.XYZ_EULER` (proper rotation-matrix composition) give sane results
  compared to the naive per-component addition `ActionType.NON_EEF` used to do, using real captured
  `ee_state`/`ee_action` data. `validate_eef_composition.py` builds `EndEffectorPose` objects
  directly; `validate_eef_xyz_euler_pipeline.py` goes through `StateActionProcessor` itself, so it
  also exercises the real `XYZ_EULER` format end-to-end, not just the composition math in isolation.

## Why these are separate from the dataset cache

`meta/modality.json` and `meta/relative_stats.json` live in the dataset cache
(`lerobot_cache/BitRobot/...`), not in this repo, and aren't tracked anywhere themselves -- a fresh
re-download would silently lose the per-arm joint groups and their stats. `patch_modality_json.py`
and `generate_relative_stats_fast.py` are what make that reproducible from this repo alone.
