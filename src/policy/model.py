from typing import Union
import torch
import torch.nn as nn

from src.policy.unet import ConditionalUnet1D
from src.policy.utils import get_resnet, replace_bn_with_gn


def get_action_diffusion_model(
        device: Union[str, torch.device] = "cpu",
        encoder_name: str = "resnet18",
        vision_feature_dim: int = 512,
        lowdim_obs_dim: int = 2,
        action_dim: int = 2,
        obs_horizon: int = 2,
):
    # construct ResNet18 encoder
    # if you have multiple camera views, use seperate encoder weights for each view.
    vision_encoder = get_resnet(encoder_name)

    # IMPORTANT!
    # replace all BatchNorm with GroupNorm to work with EMA
    # performance will tank if you forget to do this!
    vision_encoder = replace_bn_with_gn(vision_encoder)

    # ResNet18 has output dim of 512
    # agent_pos is 2 dimensional
    # observation feature has 514 dims in total per step
    obs_dim = vision_feature_dim + lowdim_obs_dim

    # create network object
    # action dim is 2 by default
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon
    )

    # the final arch has 2 parts
    nets = nn.ModuleDict({
        'vision_encoder': vision_encoder,
        'noise_pred_net': noise_pred_net
    })

    # device transfer
    device = torch.device(device)
    _ = nets.to(device)
    return nets


def load_checkpoint(model: nn.Module, checkpoint_path: str):
    """
    Load the model state from a checkpoint file.
    
    :param model: The model to load the state into.
    :param checkpoint_path: Path to the checkpoint file.
    """
    if not torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    else:
        checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {checkpoint_path}")

