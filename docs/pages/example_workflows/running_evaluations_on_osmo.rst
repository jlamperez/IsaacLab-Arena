Running Large-scale Evaluations on OSMO
=======================================

Evaluating a policy over many environments and episodes is time-consuming on a single
machine. Arena uses `NVIDIA OSMO <https://developer.nvidia.com/osmo>`__, a cloud-native
orchestration platform, to run evaluations on multi-node GPU clusters. Arena packages an
evaluation — the policy runner and its co-scheduled inference server — as OSMO workflows
and submits them.

.. note::

  For now these instructions apply to NVIDIA employees only: they assume access to an
  internal OSMO cluster and to the container images and credentials it uses.

Prerequisites
-------------

* **OSMO CLI.** Install the CLI and log in to your OSMO cluster; see the
  `OSMO documentation <https://developer.nvidia.com/osmo>`__ and your cluster's
  documentation portal (``https://<your-osmo-cluster>/docs``) for instructions.

* **Credentials.** The workflows expect the following `OSMO credentials
  <https://developer.nvidia.com/osmo>`__ to be registered once per account with
  ``osmo credential set``:

  * ``omni_svc`` — GENERIC credential with ``omni_user`` and ``omni_pass`` fields,
    used to fetch assets from Omniverse.
  * A DATA credential for ``swift://pdx.s8k.io/AUTH_team-isaac``, where evaluation
    outputs are uploaded.

Submitting an evaluation (single environment)
---------------------------------------------

``osmo.submit_evaluation_workflow`` evaluates one policy in one environment. Select the
policy with ``--policy`` (``pi0`` or ``gr00t``) and pass the Arena environment and its
arguments:

.. code-block:: bash

  python -m osmo.submit_evaluation_workflow \
      --policy gr00t \
      --arena_env kitchen_pick_and_place \
      --arena_env_args '--object cracker_box --embodiment franka_ik' \
      --policy_runner_args '--num_episodes 5 --enable_cameras --record_camera_video'

Add ``--dry_run`` to print the rendered workflow YAML without submitting. See
``python -m osmo.submit_evaluation_workflow --policy <policy> --help`` for the full
set of options (pools, images, per-task arguments), and the module docstring of
``osmo/submit_evaluation_workflow.py`` for per-policy examples.

Each submission prints the workflow ID and an overview URL for monitoring progress.
Evaluation results (metrics and recorded videos) are uploaded to
``swift://pdx.s8k.io/AUTH_team-isaac/isaaclab_arena/workflows/<workflow_id>``.

Submitting a multi-environment evaluation
-----------------------------------------

``osmo.submit_arena_experiment`` evaluates a whole Arena Experiment — many environments,
and optionally many policies, in one submission. It takes a typed Experiment YAML and
fans its Runs out across the cluster:
each Run becomes an independently scheduled OSMO group, and the inference server for a
Run is derived from that Run's policy, so an Experiment can mix pi0, GR00T, and Cosmos
Runs. The per-Run outputs are collected into a single Experiment output at the end.

.. code-block:: bash

  python -m osmo.submit_arena_experiment \
      --experiment_cfg isaaclab_arena_environments/experiment_configs/droid_pnp_srl_openpi_experiment.yaml \
      osmo.workflow_name=my-evaluation

Any trailing ``KEY=VALUE`` argument is a Hydra override applied to the composed
submission, so the Experiment can be adjusted without editing its YAML:

.. code-block:: bash

  # Run four episodes per Run and deploy the pi0 server with a different variant.
  python -m osmo.submit_arena_experiment \
      --experiment_cfg <experiment>.yaml \
      experiment_cfg.runs.<run_name>.rollout_limit.num_episodes=4 \
      servers.pi0.policy_variant=pi0

Overrides are applied in the order typed defaults < Experiment YAML < CLI overrides.
Two flags help before committing to a submission:

* ``--list_overrides`` prints the fully composed submission; every leaf in that output
  is a valid ``KEY=VALUE`` override.
* ``--dry_run`` renders the workflow YAML that would be submitted, without submitting it.
