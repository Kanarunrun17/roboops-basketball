import torch
from lerobot.policies.act.modeling_act import ACTPolicy

from roboops.policy import build_act_policy

MODEL_CONFIG = {
    "chunk_size": 8,
    "dim_model": 32,
    "n_heads": 4,
    "dim_feedforward": 64,
    "n_encoder_layers": 1,
    "n_decoder_layers": 1,
    "use_vae": False,
    "temporal_ensemble_coeff": 0.01,
    "dropout": 0.0,
}


def test_official_act_can_be_instantiated() -> None:
    policy = build_act_policy(MODEL_CONFIG)
    assert isinstance(policy, ACTPolicy)
    assert policy.config.dim_model == 32


def test_official_act_can_be_saved_and_reloaded(tmp_path) -> None:
    policy = build_act_policy(MODEL_CONFIG)
    observation = {
        "observation.state": torch.zeros(1, 2),
        "observation.environment_state": torch.zeros(1, 16),
    }
    policy.save_pretrained(tmp_path)
    reloaded = ACTPolicy.from_pretrained(tmp_path)
    with torch.inference_mode():
        actions = reloaded.predict_action_chunk(observation)
    assert actions.shape == (1, 8, 2)
    assert torch.isfinite(actions).all()
