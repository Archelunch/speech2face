from glow.model import Glow

def get_frozen_glow(cfg):
        model = Glow(
        cfg.image_shape,
        cfg.hidden_channels,
        cfg.K,
        cfg.L,
        cfg.actnorm_scale,
        cfg.flow_permutation,
        cfg.flow_coupling,
        cfg.LU_decomposed,
        cfg.learn_top,
        cfg.y_condition,
    )

    for param in model.bert.parameters():
        param.requires_grad = False

    return model
