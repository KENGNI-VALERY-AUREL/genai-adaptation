from src.utils import set_seed, get_device, load_config

cfg = load_config()
set_seed(cfg["seed"])
print("device :", get_device())
print("Model   :", cfg["model"]["name"])

import torch
print("torch    :", torch.__version__)