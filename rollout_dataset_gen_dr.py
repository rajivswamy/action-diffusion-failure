import numpy as np
import pandas as pd
import os
import json

# data imports
from  src.dataset import get_dataset_pusht
# model imports
from src.policy import get_action_diffusion_model, load_checkpoint 
# env import
from src.envs.sim_pusht import PushTImageEnv
from src.envs.sim_pusht.pusht_env_simple import PushTImageEnvSimple
# rollout import
from src.rollout import rollout

from skvideo.io import vwrite


# Global config variables
dataset_path = 'demo/pusht_cchi_v7_replay.zarr.zip'
pred_horizon = 16
obs_horizon = 2
action_horizon = 8

device = "cuda:0"
encoder_name = "resnet18"
vision_feature_dim = 512
lowdim_obs_dim = 2
action_dim = 2

num_epochs = 100
batch_size = 64
num_workers = 4
num_diffusion_iters = 100

max_steps = 300
trajectory_sample_size = 256

num_trials = 50

log_dir = "logs/datasets/domain_randomization_v3"
checkpoint_path = "demo/ema_100ep_pretrained_paper.pth"

# DR Variables
scale_low=0.5
scale_high=1.0 
uniform_scaling=True

def main():
    
    os.makedirs(log_dir, exist_ok=True)

    assert os.path.exists(checkpoint_path), f"Checkpoint path {checkpoint_path} does not exist."

    config = {
        "dataset_path":         dataset_path,
        "pred_horizon":         pred_horizon,
        "obs_horizon":          obs_horizon,
        "action_horizon":       action_horizon,
        "device":               device,
        "encoder_name":         encoder_name,
        "vision_feature_dim":   vision_feature_dim,
        "lowdim_obs_dim":       lowdim_obs_dim,
        "action_dim":           action_dim,
        "num_epochs":           num_epochs,
        "batch_size":           batch_size,
        "num_workers":          num_workers,
        "num_diffusion_iters":  num_diffusion_iters,
        "max_steps":            max_steps,
        "trajectory_sample_size": trajectory_sample_size,
        "num_trials":           num_trials,
        "log_dir":              log_dir,
        "checkpoint_path":      checkpoint_path,
        "domain_randomization": {
            "scale_low": scale_low,
            "scale_high": scale_high,
            "uniform_scaling": uniform_scaling
        }
    }


    cfg_path = os.path.join(log_dir, "config.json")
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Wrote config to {cfg_path}")


    _, stats = get_dataset_pusht(dataset_path, pred_horizon, obs_horizon, action_horizon)

    nets = get_action_diffusion_model(
        device=device,
        encoder_name=encoder_name,
        vision_feature_dim=vision_feature_dim,
        lowdim_obs_dim=lowdim_obs_dim,
        action_dim=action_dim,
        obs_horizon=obs_horizon
    )

    load_checkpoint(model=nets, checkpoint_path=checkpoint_path, device=device)
    
    width = len(str(num_trials - 1))

    for i in range(num_trials):

        name = f"episode_{i:0{width}d}"

        seed = np.random.randint(210,25536)

        env = PushTImageEnv(seed=seed, 
                            scale_low=scale_low, 
                            scale_high=scale_high, 
                            uniform_scaling=uniform_scaling)

        rollout_data, success = rollout(ema_nets=nets,
            env=env,
            stats=stats,
            max_steps=max_steps,
            obs_horizon=obs_horizon,
            action_horizon=action_horizon,
            pred_horizon=pred_horizon,
            device=device,
            num_diffusion_iters=num_diffusion_iters,
            action_dim=action_dim,
            trajectory_sample_size=trajectory_sample_size)
        
        if success:
            print(f"Rollout {i} succeeded")

        suffix = "success" if success else "failure"

        imgs = sum(rollout_data["rgb"], [])
        # save video
        video_path = os.path.join(log_dir, f"{name}_{suffix}.mp4")
        vwrite(video_path, imgs)

        # save data
        df = pd.DataFrame(rollout_data)

        # add a column 'success' to the dataframe
        # fills the entire column with the value of success
        df["success"] = success
        df["episode"] = i

        df.to_pickle(os.path.join(log_dir, f"{name}_{suffix}.pkl"))

if __name__ == "__main__":
    main()
