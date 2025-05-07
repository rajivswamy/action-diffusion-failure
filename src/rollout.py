
import numpy as np
import torch
import collections
from tqdm import tqdm

from src.dataset import normalize_data, unnormalize_data
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

def rollout(ema_nets, 
            env, 
            stats, 
            max_steps = 200, 
            obs_horizon = 2, 
            action_horizon = 8, 
            pred_horizon = 16, 
            device='cuda', 
            num_diffusion_iters=100,
            action_dim=2,
            trajectory_sample_size=128):
    # assume the env is already seeded
    # get first observation
    obs, info = env.reset()

    # init noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule='squaredcos_cap_v2',
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type='epsilon'
    )

    # keep a queue of last 2 steps of observations
    obs_deque = collections.deque(
        [obs] * obs_horizon, maxlen=obs_horizon)
    # save visualization and rewards
    imgs = []
    rewards = []
    done = False
    step_idx = 0

    # data logs, ideally matched up with each time step
    timestep = [] # must be staggered by action_horizon
    executed_action = [] # elems must be action_horizon x action_dim
    sampled_actions = [] # elems must be samples x pred_horizon x action_dim
    action_index = []
    agent_positions = []
    agent_velocities = []
    block_poses = []
    goal_poses = []
    step_image_features = []

    success = False

    with tqdm(total=max_steps, desc="Eval PushTImageEnv") as pbar:
        while not done:
            # controls the batch size of trajectories sampled
            B = trajectory_sample_size
            # stack the last obs_horizon number of observations
            images = np.stack([x['image'] for x in obs_deque])
            agent_poses = np.stack([x['agent_pos'] for x in obs_deque])

            # normalize observation
            nagent_poses = normalize_data(agent_poses, stats=stats['agent_pos'])
            # images are already normalized to [0,1]
            nimages = images

            # device transfer
            nimages = torch.from_numpy(nimages).to(device, dtype=torch.float32)
            # (2,3,96,96)
            nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)
            # (2,2)

            # infer action
            with torch.no_grad():
                # get image features
                image_features = ema_nets['vision_encoder'](nimages)
                # (2,512)

                # log the image features for each replanning step
                step_image_features.append(image_features.cpu().numpy())

                # concat with low-dim observations
                obs_features = torch.cat([image_features, nagent_poses], dim=-1)

                # reshape observation to (B,obs_horizon*obs_dim)
                obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)
                obs_cond = obs_cond.expand(B, -1)  # (B, obs_horizon * obs_dim)

                # initialize action from Guassian noise
                noisy_action = torch.randn(
                    (B, pred_horizon, action_dim), device=device)
                naction = noisy_action

                # init scheduler
                noise_scheduler.set_timesteps(num_diffusion_iters)

                for k in noise_scheduler.timesteps:
                    # predict noise
                    noise_pred = ema_nets['noise_pred_net'](
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond
                    )

                    # inverse diffusion step (remove noise)
                    naction = noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction
                    ).prev_sample

            # unnormalize & slice to the first action_horizon
            naction_np    = naction.cpu().numpy()  # (B, pred_horizon, action_dim)
            pred_batch    = unnormalize_data(naction_np, stats=stats['action'])
            # log the sampled actions
            sampled_actions.append(pred_batch) # (B, pred_horizon, action_dim)
            # pick one trajectory to execute and log the sample
            idx               = np.random.randint(0, B) # randomly sampled
            executed_action.append(pred_batch[idx]) # (pred_horizon, action_dim)
            # restrict view to action horizon
            start, end    = obs_horizon - 1, (obs_horizon - 1) + action_horizon 
            action_index.append((start, end))
            executed_traj  = pred_batch[idx, start:end, :]     # (action_horizon,action_dim)

            # log timestep
            timestep.append(step_idx)

            traj_images = []
            traj_rewards = []
            traj_agent_pos = []
            traj_agent_vel = []
            traj_block_pos = []
            traj_goal_pos  = []

            # execute action_horizon number of steps
            # without replanning
            for i in range(len(executed_traj)):

                action = executed_traj[i]

                # add logs
                traj_images.append(env.render(mode='rgb_array'))
                traj_agent_pos.append(obs['agent_pos'])
                traj_agent_vel.append(info['vel_agent'])
                traj_block_pos.append(info['block_pose'])
                traj_goal_pos.append(info['goal_pose'])

                # stepping env
                obs, reward, done, _, info = env.step(action)
                # save observations
                obs_deque.append(obs)
                # and reward/vis
                traj_rewards.append(reward)
                
                if reward > 0.999:
                    success = True

                # update progress bar
                step_idx += 1
                pbar.update(1)
                pbar.set_postfix(reward=reward)
                if step_idx > max_steps:
                    done = True
                if done:
                    break

            # add logs 
            imgs.append(traj_images)
            agent_positions.append(traj_agent_pos)
            agent_velocities.append(traj_agent_vel)
            block_poses.append(traj_block_pos)
            goal_poses.append(traj_goal_pos)
            rewards.append(traj_rewards)


    data = {
        'timestep': timestep,
        'rgb': imgs,
        'reward': rewards,
        'sampled_actions': sampled_actions,
        'executed_action': executed_action,
        'action_index': action_index,
        'agent_positions': agent_positions,
        'agent_velocities': agent_velocities,
        'block_poses': block_poses,
        'goal_poses': goal_poses,
        'step_image_features': step_image_features,
    }

    return data, success
