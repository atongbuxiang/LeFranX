from pathlib import Path

# Canonical feature keys
ACTION = "action"
OBS_STATE = "observation.state"
OBS_ENV_STATE = "observation.environment_state"
OBS_IMAGE = "observation.image"
OBS_IMAGES = "observation.images"
REWARD = "next.reward"

# Local cache/layout paths
HF_LEROBOT_HOME = Path.home() / ".cache" / "lerobot"
HF_LEROBOT_CALIBRATION = HF_LEROBOT_HOME / "calibration"
ROBOTS = "robots"

# Checkpoint/layout filenames
CHECKPOINTS_DIR = "checkpoints"
LAST_CHECKPOINT_LINK = "last"
PRETRAINED_MODEL_DIR = "pretrained_model"
TRAINING_STATE_DIR = "training_state"
TRAINING_STEP = "training_step.json"
OPTIMIZER_PARAM_GROUPS = "optimizer_param_groups.json"
OPTIMIZER_STATE = "optimizer_state.safetensors"
RNG_STATE = "rng_state.safetensors"
SCHEDULER_STATE = "scheduler_state.json"
