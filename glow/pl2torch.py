import torch
from model import Glow
from omegaconf import OmegaConf
from collections import OrderedDict


model_path = '/home/mvpavlukhin/speech2face/output/model15.ckpt'
new_torch_model_path = '/home/mvpavlukhin/speech2face/output/glow_torch.cpkt'
config_path = '../config.yaml'

if __name__ == "__main__":
    checkpoint = torch.load(model_path, map_location='cpu')
    config = OmegaConf.load(config_path)
    print('im shape',config.glow.image_shape)
    model = Glow(**config.glow)

    keys_orig = list(model.state_dict().keys())
    keys_pl =  list(checkpoint['state_dict'].keys())

    new_state_dict = OrderedDict()

    for key, value in checkpoint['state_dict'].items():
        new_key = key.replace('model.','')
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    torch.save(model.state_dict(), new_torch_model_path)
