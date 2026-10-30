from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class MicroduckPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Configuration for PPO policy training on Microduck biped locomotion."""

    num_steps_per_env: int = 24
    max_iterations: int = 10000
    save_interval: int = 100
    experiment_name: str = "microduck_velocity"
    empirical_normalization: bool = False
    clip_actions: float = 1.0

    obs_groups: dict[str, list[str]] = {
        "actor": ["policy"],
        "critic": ["policy"],
    }

    actor: RslRlMLPModelCfg = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )

    critic: RslRlMLPModelCfg = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )

    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

