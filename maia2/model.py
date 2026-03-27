import gdown
import os
from .main import MAIA2Model
from .utils import get_all_possible_moves, create_elo_dict, parse_args
import torch
from torch import nn
import warnings
warnings.filterwarnings("ignore")
import pdb

def from_pretrained(type, device, save_root = "./maia2_models"):
    
    output_path = os.path.join(save_root, "rapid_model.pt")

    if not os.path.exists(output_path):
        raise ValueError("Model at " + output_path + " does not exist")

    cfg_path = os.path.join(save_root, "config.yaml")
    if not os.path.exists(cfg_path):
        raise ValueError("Config does not exist")

    cfg = parse_args(cfg_path)

    all_moves = get_all_possible_moves()
    elo_dict = create_elo_dict()

    model = MAIA2Model(len(all_moves), elo_dict, cfg)
    model = nn.DataParallel(model)
    
    checkpoint = torch.load(output_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.module
    
    if device == "gpu":
        model = model.cuda()
    
    print(f"Model for {type} games loaded to {device}.")
    
    return model
    
    
    
    
    
    
