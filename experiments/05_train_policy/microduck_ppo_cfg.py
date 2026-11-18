from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

from microduck_env_cfg import mirror_symmetry_augmentation


def build_runner(wrapped_env, agent_cfg: "MicroduckPPORunnerCfg", log_dir: str | None, device: str):
    """Create the rsl_rl OnPolicyRunner for this config.

    Works across Isaac Lab versions: the config clean-up helpers are used when the installed
    ``isaaclab_rl`` still ships them and skipped when it does not.
    """
    from rsl_rl.runners import OnPolicyRunner

    try:
        from isaaclab_rl.rsl_rl import check_rsl_rl_version, handle_deprecated_rsl_rl_cfg

        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    except ImportError:
        pass
    return OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=log_dir, device=device)


@configclass
class MicroduckPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Configuration for PPO policy training on Microduck biped locomotion."""

    num_steps_per_env: int = 24
    max_iterations: int = 10000
    save_interval: int = 200
    experiment_name: str = "microduck_agile_v2"
    empirical_normalization: bool = False
    # Wide clip on purpose. With a tight +/-1 clip the environment never sees large actions, so the action
    # penalties cannot push back on the exploration noise and the entropy bonus inflates it without limit
    # (the previous run ended with a noise std of ~30). play.py must use the same value.
    clip_actions: float = 3.0

    obs_groups: dict[str, list[str]] = {
        "actor": ["policy"],
        "critic": ["policy", "critic"],
    }

    actor: RslRlMLPModelCfg = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        # std_range is a safety net: a healthy run settles between the bounds. The lower bound keeps
        # exploration alive (the noise fell to 0.23 within 200 iterations in validation).
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.6, std_type="log", std_range=(0.15, 1.5)
        ),
    )

    critic: RslRlMLPModelCfg = RslRlMLPModelCfg(
        hidden_dims=[512, 512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )

    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # Left/right symmetry: every mini-batch is trained on both the samples and their mirror images,
        # so the policy behaves the same way turning or strafing left as it does right.
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=False,
            data_augmentation_func=mirror_symmetry_augmentation,
        ),
    )

