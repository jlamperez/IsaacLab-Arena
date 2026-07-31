# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Registry binding each remote client policy to the OSMO inference server that serves it.

The Arena-experiment workflow derives, per Run, which server to co-schedule from the Run's
client policy config type. Each binding declares the server task to launch, the resource it
needs, and any per-server compatibility check.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

from isaaclab_arena.assets.registries import Registry
from isaaclab_arena.evaluation.arena_experiment import ArenaExperimentCfg
from isaaclab_arena.evaluation.arena_run import ArenaRunCfg
from isaaclab_arena.policy.policy_base import PolicyCfg
from isaaclab_arena_cosmos.policy.cosmos_remote_config import CosmosRemotePolicyCfg
from isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy import Gr00tRemoteClosedloopPolicyCfg
from isaaclab_arena_openpi.policy.pi0_remote_config import Pi0RemotePolicyCfg
from osmo.tasks.base_task import BaseTask, TaskCfg
from osmo.tasks.cosmos_server_task import CosmosServerTask, CosmosServerTaskCfg
from osmo.tasks.gr00t_server_task import Gr00tServerTask, Gr00tServerTaskCfg
from osmo.tasks.pi0_server_task import Pi0ServerTask, Pi0ServerTaskCfg
from osmo.workflows.workflow import WorkflowCfg
from osmo.workflows.workflow_constants import POLICY_SERVER_PORT

# Servers default to the same resource the workflow uses today; a server type that needs
# different hardware overrides these, and a submission requires all of its servers to agree.
_DEFAULT_SERVER_POOL = WorkflowCfg().pool
_DEFAULT_SERVER_PLATFORM = WorkflowCfg().platform


class ServerBinding:
    """Bind one remote client policy to the OSMO inference server that serves it."""

    name: str
    """Server-type key: the ``servers.<name>`` config field and OSMO task-name stem."""

    policy_cfg_type: type[PolicyCfg]
    """Client policy configuration served by this server."""

    server_task_cls: type[BaseTask]
    """OSMO task that serves this policy."""

    pool: str = _DEFAULT_SERVER_POOL
    """OSMO pool this server must run in."""

    platform: str = _DEFAULT_SERVER_PLATFORM
    """Hardware platform this server must run on."""

    @classmethod
    def check_runs(cls, runs_using_binding: list[ArenaRunCfg], server_cfg: TaskCfg) -> None:
        """Validate the Runs this server serves against its deployment config; no-op by default."""


class ServerBindingRegistry(Registry):
    """Registry of the OSMO inference servers available to Arena Experiment Runs."""

    def __init__(self):
        super().__init__()
        self._bindings_by_policy_cfg_type: dict[type[PolicyCfg], type[ServerBinding]] = {}

    def register_server_binding(self, binding: type[ServerBinding]) -> None:
        """Register a binding under its name and the client policy config it serves."""
        assert (
            binding.policy_cfg_type not in self._bindings_by_policy_cfg_type
        ), f"Policy config {binding.policy_cfg_type.__name__} already has a server binding"
        self.register(binding, binding.name)
        self._bindings_by_policy_cfg_type[binding.policy_cfg_type] = binding

    def get_binding_for_policy_cfg(self, policy_cfg: PolicyCfg) -> type[ServerBinding] | None:
        """Get the server for a Run's client policy, or None if that policy needs no server."""
        return self._bindings_by_policy_cfg_type.get(type(policy_cfg))

    def get_binding_by_name(self, name: str) -> type[ServerBinding]:
        """Get a server by its ``servers.<name>`` key."""
        return self.get_component_by_name(name)


def register_server_binding(cls: type[ServerBinding]) -> type[ServerBinding]:
    """Decorator registering a server binding with the ServerBindingRegistry."""
    if ServerBindingRegistry().is_registered(cls.name, ensure_loaded=False):
        print(f"WARNING: Server binding {cls.name} is already registered. Doing nothing.")
    else:
        ServerBindingRegistry().register_server_binding(cls)
    return cls


@register_server_binding
class Pi0ServerBinding(ServerBinding):
    """Serve pi0 (openpi) Runs."""

    name = "pi0"
    policy_cfg_type = Pi0RemotePolicyCfg
    server_task_cls = Pi0ServerTask

    @classmethod
    def check_runs(cls, runs_using_binding: list[ArenaRunCfg], server_cfg: TaskCfg) -> None:
        """Require every served Run to request the variant the server deploys."""
        assert isinstance(server_cfg, Pi0ServerTaskCfg)
        incompatible_policy_variants_by_run = {
            run_cfg.name: run_cfg.policy.policy_variant
            for run_cfg in runs_using_binding
            if run_cfg.policy.policy_variant != server_cfg.policy_variant
        }
        assert not incompatible_policy_variants_by_run, (
            f"pi0_remote Runs require variants {incompatible_policy_variants_by_run}, but the pi0 server is configured"
            f" for '{server_cfg.policy_variant}'"
        )


@register_server_binding
class Gr00tServerBinding(ServerBinding):
    """Serve GR00T Runs."""

    name = "gr00t"
    policy_cfg_type = Gr00tRemoteClosedloopPolicyCfg
    server_task_cls = Gr00tServerTask


@register_server_binding
class CosmosServerBinding(ServerBinding):
    """Serve Cosmos Runs."""

    name = "cosmos"
    policy_cfg_type = CosmosRemotePolicyCfg
    server_task_cls = CosmosServerTask


def configure_client_for_server(
    policy_cfg: PolicyCfg,
    binding: type[ServerBinding],
    server_task_name: str,
) -> None:
    """Point a Run's client policy at its dedicated OSMO server task (mutates the policy config)."""
    policy_cfg.remote_host = binding.server_task_cls.host_token(server_task_name)
    policy_cfg.remote_port = POLICY_SERVER_PORT


def runs_by_binding(experiment_cfg: ArenaExperimentCfg) -> dict[type[ServerBinding], list[ArenaRunCfg]]:
    """Group the Runs that need a server by the server binding that serves them."""
    registry = ServerBindingRegistry()
    grouped_runs: dict[type[ServerBinding], list[ArenaRunCfg]] = {}
    for run_cfg in experiment_cfg.runs.values():
        binding = registry.get_binding_for_policy_cfg(run_cfg.policy)
        if binding is not None:
            grouped_runs.setdefault(binding, []).append(run_cfg)
    return grouped_runs


def required_server_resource(experiment_cfg: ArenaExperimentCfg) -> tuple[str, str] | None:
    """Return the single ``(pool, platform)`` every server the Experiment needs must run on.

    Args:
        experiment_cfg: Experiment whose Runs determine the servers to co-schedule.

    Returns:
        The required resource, or None when no Run needs a server.
    """
    required_resources = set()
    for binding in runs_by_binding(experiment_cfg):
        required_resources.add((binding.pool, binding.platform))
    assert len(required_resources) <= 1, (
        f"Experiment needs servers on different resources {sorted(required_resources)}; a submission runs on a"
        " single pool. Run the incompatible policies as separate submissions."
    )
    return next(iter(required_resources), None)


@dataclass
class ServersCfg:
    """Per server-type config, shared by every server of that type in an Experiment.

    Every server type is always present with defaults so its ``servers.<name>.*`` overrides
    compose under Hydra (a ``None`` sub-config would carry no schema). Which servers actually
    launch is decided per Run from the Run's client policy type, not from these fields.
    """

    pi0: Pi0ServerTaskCfg = field(default_factory=Pi0ServerTaskCfg)
    gr00t: Gr00tServerTaskCfg = field(default_factory=Gr00tServerTaskCfg)
    cosmos: CosmosServerTaskCfg = field(default_factory=CosmosServerTaskCfg)


# ServersCfg cannot be built from the registry (Hydra needs static fields), so keep the two in step.
assert {server_field.name for server_field in fields(ServersCfg)} == set(
    ServerBindingRegistry().get_all_keys()
), "ServersCfg must declare one field per registered server binding, named after its binding"
