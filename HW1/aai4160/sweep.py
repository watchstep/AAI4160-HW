import wandb
import time
from aai4160.config import parse_args
from aai4160.scripts.run_hw1 import run_bc
from aai4160.infrastructure.utils import MJ_ENV_KWARGS, MJ_ENV_NAMES
args = parse_args()

exp_date = time.strftime("%m-%d_%H-%M-%S")
sweep_config = {
    "method": "random",
    "name": f"sweep_{exp_date}",
    "metric": {"goal": "maximize", "name": "Eval_AverageReturn"},
    "parameters": {
        "num_agent_train_steps_per_iter": {"values": [1000, 1500, 2000, 25000, 3000]},
        "batch_size": {"values":[1000, 1500, 2000, 25000, 3000]},
        "train_batch_size": {"values": [80, 100, 120, 140, 160, 180, 200]},
        "n_layers":{"values": [2, 8, 16, 32, 64]},
        "size":{"values": [16, 32, 64, 128, 256]},
        "learning_rate":{"max": 1e-1, "min": 1e-4},
        "critertion":{"values": ["MSE", "L1", "SmoothL1"]}
    },
}

sweep_id = wandb.sweep(sweep=sweep_config, project="aai4160_hw1")
wandb.agent(sweep_id,
			function=run_bc,
            entity='watchstep', 
            project='aai4160_hw1')