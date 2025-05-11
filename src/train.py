import copy
import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm import tqdm
import os

from src.dataset import PushTImageDataset
from src.ema import EMAModelNew


def train_diff_model(nets, 
                     dataset: PushTImageDataset, 
                     logdir: str,
                     save_name: str = 'model',
                     num_epochs = 2000,
                     batch_size = 64,
                     num_workers = 4,
                     device = 'cuda',
                     num_diffusion_iters = 100,
                     obs_horizon = 2,
                     checkpoint_every = 200,
                     checkpoint = None,
                     ):
    
    # make log directory if it does not exist
    os.makedirs(logdir, exist_ok=True)

    noise_pred_net = nets['noise_pred_net']

    # create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process afte each epoch
        persistent_workers=True
    )

    # create noise scheduler
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

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(
        parameters=nets.parameters(),
        power=0.75)

    # Standard ADAM optimizer
    # Note that EMA parametesr are not optimized
    optimizer = torch.optim.AdamW(
        params=nets.parameters(),
        lr=1e-4, weight_decay=1e-6)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs
    )
    
    start_epoch = 0
    if checkpoint is not None:
        ckpt = torch.load(checkpoint['path'], map_location=device)
        start_epoch = checkpoint.get('num_epochs', 100)
        print(f"✔️  Loaded model weights from {checkpoint['path']}")

        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'lr_scheduler_state_dict' in ckpt:
            lr_scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
        else:
            # manually advance scheduler to the correct step
            steps_done = start_epoch * len(dataloader)
            for _ in range(steps_done):
                lr_scheduler.step()
    else:
        print("⏳  No checkpoint provided, training from scratch")

    all_epoch_losses = []

    with tqdm(range(start_epoch,num_epochs), desc='Epoch') as tglobal:
        # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = list()
            # batch loop
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    # data normalized in dataset
                    # device transfer
                    nimage = nbatch['image'][:,:obs_horizon].to(device)
                    nagent_pos = nbatch['agent_pos'][:,:obs_horizon].to(device)
                    naction = nbatch['action'].to(device)
                    B = nagent_pos.shape[0]

                    # encoder vision features
                    image_features = nets['vision_encoder'](
                        nimage.flatten(end_dim=1))
                    image_features = image_features.reshape(
                        *nimage.shape[:2],-1)
                    # (B,obs_horizon,D)

                    # concatenate vision feature and low-dim obs
                    obs_features = torch.cat([image_features, nagent_pos], dim=-1)
                    obs_cond = obs_features.flatten(start_dim=1)
                    # (B, obs_horizon * obs_dim)

                    # sample noise to add to actions
                    noise = torch.randn(naction.shape, device=device)

                    # sample a diffusion iteration for each data point
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (B,), device=device
                    ).long()

                    # add noise to the clean images according to the noise magnitude at each diffusion iteration
                    # (this is the forward diffusion process)
                    noisy_actions = noise_scheduler.add_noise(
                        naction, noise, timesteps)

                    # predict the noise residual
                    noise_pred = noise_pred_net(
                        noisy_actions, timesteps, global_cond=obs_cond)

                    # L2 loss
                    loss = nn.functional.mse_loss(noise_pred, noise)

                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # step lr scheduler every batch
                    # this is different from standard pytorch behavior
                    lr_scheduler.step()

                    # update Exponential Moving Average of the model weights
                    ema.step(nets.parameters())

                    # logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
            tglobal.set_postfix(loss=np.mean(epoch_loss))
            all_epoch_losses.append(np.mean(epoch_loss))

            # save EMA params of model every checkpoint_every epochs
            if (epoch_idx + 1) % checkpoint_every == 0:
                # deep‐copy nets so we don’t overwrite your “live” training model
                ema_nets_cp = copy.deepcopy(nets)
                # copy EMA weights into that copy
                ema.copy_to(ema_nets_cp.parameters())
                path_name = os.path.join(logdir, f"checkpoint_ema_epoch_{epoch_idx+1}.pth")
                torch.save({'model_state_dict': ema_nets_cp.state_dict()}, 
                    path_name)
                

    # Weights of the EMA model
    # is used for inference
    ema_nets = nets
    ema.copy_to(ema_nets.parameters())

    final_path = os.path.join(logdir, f"{save_name}_ema_epoch_{num_epochs}.pth")

    # save loss data and model to the logdir
    torch.save({'model_state_dict': ema_nets.state_dict()}, 
               final_path)
    np.save(os.path.join(logdir, 'loss.npy'), np.array(all_epoch_losses))

    print(f"Training finished. Model saved to {logdir}/model.pth")

    return ema_nets, np.array(epoch_loss)

def train_diff_model_new_ema(nets, 
                     dataset: PushTImageDataset, 
                     logdir: str,
                     save_name: str = 'model',
                     num_epochs = 2000,
                     batch_size = 64,
                     num_workers = 4,
                     device = 'cuda',
                     num_diffusion_iters = 100,
                     obs_horizon = 2,
                     checkpoint_every = 200,
                     checkpoint = None
                     ):
    
    # make log directory if it does not exist
    os.makedirs(logdir, exist_ok=True)

    noise_pred_net = nets['noise_pred_net']

    # create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process afte each epoch
        persistent_workers=True
    )

    # create noise scheduler
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

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModelNew(
        model=copy.deepcopy(nets),
        update_after_step=0,     # start EMA from step 0
        inv_gamma=1.0,
        power=0.75,              # same “power” schedule you had before
        min_value=0.0,
        max_value=0.9999,
    )
    ema.averaged_model.to(device)


    # Standard ADAM optimizer
    # Note that EMA parametesr are not optimized
    optimizer = torch.optim.AdamW(
        params=nets.parameters(),
        lr=1e-4, weight_decay=1e-6)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs
    )

    start_epoch = 0
    if checkpoint is not None:
        ckpt = torch.load(checkpoint['path'], map_location=device)
        start_epoch = checkpoint['num_epochs']
        print(f"✔️  Loaded model weights from {checkpoint['path']}")

        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'lr_scheduler_state_dict' in ckpt:
            lr_scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
        else:
            # manually advance scheduler to the correct step
            steps_done = start_epoch * len(dataloader)
            for _ in range(steps_done):
                lr_scheduler.step()
    else:
        print("⏳  No checkpoint provided, training from scratch")

    all_epoch_losses = []

    with tqdm(range(start_epoch, num_epochs), desc='Epoch') as tglobal:
        # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = list()
            # batch loop
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    # data normalized in dataset
                    # device transfer
                    nimage = nbatch['image'][:,:obs_horizon].to(device)
                    nagent_pos = nbatch['agent_pos'][:,:obs_horizon].to(device)
                    naction = nbatch['action'].to(device)
                    B = nagent_pos.shape[0]

                    # encoder vision features
                    image_features = nets['vision_encoder'](
                        nimage.flatten(end_dim=1))
                    image_features = image_features.reshape(
                        *nimage.shape[:2],-1)
                    # (B,obs_horizon,D)

                    # concatenate vision feature and low-dim obs
                    obs_features = torch.cat([image_features, nagent_pos], dim=-1)
                    obs_cond = obs_features.flatten(start_dim=1)
                    # (B, obs_horizon * obs_dim)

                    # sample noise to add to actions
                    noise = torch.randn(naction.shape, device=device)

                    # sample a diffusion iteration for each data point
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (B,), device=device
                    ).long()

                    # add noise to the clean images according to the noise magnitude at each diffusion iteration
                    # (this is the forward diffusion process)
                    noisy_actions = noise_scheduler.add_noise(
                        naction, noise, timesteps)

                    # predict the noise residual
                    noise_pred = noise_pred_net(
                        noisy_actions, timesteps, global_cond=obs_cond)

                    # L2 loss
                    loss = nn.functional.mse_loss(noise_pred, noise)

                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # step lr scheduler every batch
                    # this is different from standard pytorch behavior
                    lr_scheduler.step()

                    # update Exponential Moving Average of the model weights
                    ema.step(nets)

                    # logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
            tglobal.set_postfix(loss=np.mean(epoch_loss))
            all_epoch_losses.append(np.mean(epoch_loss))

            # save EMA params of model every checkpoint_every epochs
            if (epoch_idx + 1) % checkpoint_every == 0:
                ema_nets_cp = copy.deepcopy(nets)
                ema_nets_cp.load_state_dict(ema.averaged_model.state_dict())
                path = os.path.join(logdir, f"checkpoint_ema_epoch_{epoch_idx+1}.pth")
                torch.save({'model_state_dict': ema_nets_cp.state_dict()}, path)
                

    # Weights of the EMA model
    # is used for inference
    ema_nets = ema.averaged_model

    final_path = os.path.join(logdir, f"{save_name}_ema_epoch_{num_epochs}.pth")

    # save loss data and model to the logdir
    torch.save({'model_state_dict': ema_nets.state_dict()}, 
               final_path)
    np.save(os.path.join(logdir, 'loss.npy'), np.array(all_epoch_losses))

    print(f"Training finished. Model saved to {logdir}/model.pth")

    return ema_nets, np.array(epoch_loss)
