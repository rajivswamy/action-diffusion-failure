# data imports
from  src.dataset import get_dataset_pusht
# model imports
from src.policy import get_action_diffusion_model 
# train import
from src.train import train_diff_model

import os

dataset_path = 'demo/pusht_cchi_v7_replay.zarr.zip'
pred_horizon = 16
obs_horizon = 2
action_horizon = 8

device = "cuda:0"
encoder_name = "resnet18"
vision_feature_dim = 512
lowdim_obs_dim = 2
action_dim = 2

num_epochs = 300
batch_size = 256
num_workers = 4
num_diffusion_iters = 100
checkpoint_every = 100

logdir = "logs/train/default_long_run"
model_name = "action-diff"

def main():
    os.makedirs(logdir, exist_ok=True)

    dataset, stats = get_dataset_pusht(dataset_path, pred_horizon, obs_horizon, action_horizon)

    nets = get_action_diffusion_model(
        device=device,
        encoder_name=encoder_name,
        vision_feature_dim=vision_feature_dim,
        lowdim_obs_dim=lowdim_obs_dim,
        action_dim=action_dim,
        obs_horizon=obs_horizon
    )

    nets, losses = train_diff_model(
        nets=nets,
        dataset=dataset,
        logdir=logdir,
        save_name=model_name,
        num_epochs=num_epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        num_diffusion_iters=num_diffusion_iters,
        obs_horizon=obs_horizon,
        checkpoint_every=checkpoint_every
    )

if __name__ == "__main__":
    main()
