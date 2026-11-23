"""
Distributed learning framework based on Multi-agent Proximal Policy Optimization
Created by Yifeng Zhang
"""
import numpy as np
import random

import torch.nn.functional as F

from utils import *
from ppo_clipping import PolicyClipping
from runner_unicorn import Runner
from models.Unicorn import Unicorn
from torch.distributions.categorical import Categorical



def write_to_Tensorboard(global_summary, tensorboard_data, curr_episode, plot_means=True):
    # each row in tensorboardData represents an episode
    # each column is a specific metric
    if plot_means:
        tensorboard_data = np.array(tensorboard_data)
        tensorboard_data = list(np.mean(tensorboard_data, axis=0))
        (policy_loss, value_loss, entropy_loss, a_grad_norm, c_grad_norm, clip_frac, a_vae_loss, c_vae_loss, a_c_loss, c_c_loss,
            episode_reward, episode_length, action_change) = tensorboard_data

    else:
        first_episode = tensorboard_data[0]
        (policy_loss, value_loss, entropy_loss, a_grad_norm, c_grad_norm, clip_frac, a_vae_loss, c_vae_loss, a_c_loss, c_c_loss,
         episode_reward, episode_length, action_change) = first_episode

    global_summary.add_scalar(tag='Losses/Value Loss', scalar_value=value_loss, global_step=curr_episode)
    global_summary.add_scalar(tag='Losses/Policy Loss', scalar_value=policy_loss, global_step=curr_episode)
    global_summary.add_scalar(tag='Losses/Entropy Loss', scalar_value=entropy_loss, global_step=curr_episode)
    global_summary.add_scalar(tag='Losses/Actor VAE Loss', scalar_value=a_vae_loss, global_step=curr_episode)
    global_summary.add_scalar(tag='Losses/Critic VAE Loss', scalar_value=c_vae_loss, global_step=curr_episode)
    global_summary.add_scalar(tag='Losses/Actor Contrastive Loss', scalar_value=a_c_loss, global_step=curr_episode)
    global_summary.add_scalar(tag='Losses/Critic Contrastive Loss', scalar_value=c_c_loss, global_step=curr_episode)
    global_summary.add_scalar(tag='Losses/Actor Grad Norm', scalar_value=a_grad_norm, global_step=curr_episode)
    global_summary.add_scalar(tag='Losses/Critic Grad Norm', scalar_value=c_grad_norm, global_step=curr_episode)
    global_summary.add_scalar(tag='Losses/Clip Fraction', scalar_value=clip_frac, global_step=curr_episode)

    global_summary.add_scalar(tag='Perf/Episode Reward', scalar_value=episode_reward, global_step=curr_episode)
    global_summary.add_scalar(tag='Perf/Episode Length', scalar_value=episode_length, global_step=curr_episode)
    global_summary.add_scalar(tag='Perf/Action Change', scalar_value=action_change, global_step=curr_episode)


def get_global_train_buffer(all_jobs):
    """
    Combine experience buffers from all workers:
        1. Batch observations (torch.tensor)
        2. Batch actions (torch.tensor)
        3. Batch log prob old (torch.tensor)
        4. Batch phase mask (torch.tensor)
        5. Batch phase vector (torch.tensor)
        6. Batch intersection attribute vector (torch.tensor)
        7. Batch next observations for prediction (torch.tensor)
        8. Batch neighbor actions vector (torch.tensor)
        9. Batch advantages (numpy array)
        10. Batch target values (numpy array)
        Additional full sampled policy tensor for dppo_full_tv only.
    """
    buffer_len = 10 + int(NETWORK_PARAMS.CLIPPING == 'dppo_full_tv')
    global_buffer = []
    global_metrics = []
    for i in range(buffer_len):
        global_buffer.append([])
    random.shuffle(all_jobs)
    for job in all_jobs:
        job_results, metrics, info = job
        for i in range(len(global_buffer)):
            global_buffer[i].append(job_results[0][i])
        global_metrics.append(metrics)

    # cat all jobs results, agent dim from n to n * meta_agent
    for i, item in enumerate(global_buffer):
        global_buffer[i] = torch.cat(item, 1)

    global_metrics = np.mean(global_metrics, 0)

    return global_buffer, global_metrics


def calculate_gradients_ma_ppo(network, device, experience_buffers, norm_adv=True):
    co_train = NETWORK_PARAMS.CO_TRAIN
    k_epoch = NETWORK_PARAMS.K_EPOCH
    eps_clip = NETWORK_PARAMS.EPS_CLIP
    grad_clip = NETWORK_PARAMS.GRAD_CLIP
    vl_factor = NETWORK_PARAMS.VL_FACTOR
    el_factor = NETWORK_PARAMS.EL_FACTOR
    pl_factor = NETWORK_PARAMS.PL_FACTOR
    cl_factor = NETWORK_PARAMS.CL_FACTOR
    con_temp = NETWORK_PARAMS.CONTRASTIVE_TEMP
    con_batch = NETWORK_PARAMS.CONTRASTIVE_BATCH
    if co_train:
        num_meta = 1
    else:
        num_meta = TRAIN_PARAMS.NUM_META_AGENTS

    v_l, p_l, e_l, a_v_l, c_v_l, a_gn, c_gn, c_f, a_c_l, c_c_l = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    batch_obs = experience_buffers[0].to(device)
    batch_actions = experience_buffers[1].to(device)
    batch_log_p_old = experience_buffers[2].to(device)
    batch_phase_mask = experience_buffers[3].to(device)
    batch_phase_vec = experience_buffers[4].to(device)
    batch_int_attr_vec = experience_buffers[5].to(device)
    batch_target_prediction = experience_buffers[6].to(device)
    batch_neighbor_actions_vec = experience_buffers[7].to(device)

    batch_advantages = experience_buffers[8].to(device)
    batch_target_values = experience_buffers[9].to(device)
    if norm_adv:
        batch_advantages = batch_advantages - batch_advantages.mean() / (batch_advantages.std() + 1e-5)

    clipping = PolicyClipping(NETWORK_PARAMS.CLIPPING, batch_log_p_old,
                              experience_buffers[-1] if NETWORK_PARAMS.CLIPPING == 'dppo_full_tv' else None,
                              eps_clip)

    for k in range(k_epoch):
        network.actor_optimizer.zero_grad()

        policy, _, prediction, mu, logvar = network.forward(batch_obs, batch_phase_vec, batch_int_attr_vec, batch_phase_mask, None, num_meta)
        policy_dist = Categorical(policy)
        # Calculate policy loss
        log_p = policy_dist.log_prob(batch_actions)
        imp_weights = torch.exp(log_p - batch_log_p_old)
        policy_loss, clip_mask = clipping.loss(policy, imp_weights, batch_advantages)
        entropy_loss = policy_dist.entropy().mean()

        # Calculate prediction loss for Actor VAE
        prediction = prediction.gather(2, batch_actions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, prediction.size(-1))).reshape(prediction.size(0), prediction.size(1), -1)
        mu = mu.gather(2, batch_actions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, mu.size(-1))).reshape(mu.size(0), mu.size(1), -1)
        logvar = logvar.gather(2, batch_actions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, logvar.size(-1))).reshape(logvar.size(0), logvar.size(1), -1)
        vae_loss = torch.mean(torch.square(batch_target_prediction.reshape(prediction.size()) - prediction)) - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Calculate contrastive loss for Actor network based on NT-Xent contrastive loss
        a_cl_loss = 0
        # Shape: (time_steps, num_meta, n_agents, feature_dim)
        features = mu.reshape(mu.size(0), num_meta, -1, network.actor_network.vae_hidden_dim)
        time_steps, num_meta, num_agents, feature_dim = features.size()
        # Select a random subset of 256 timesteps if there are more than 256 timesteps available
        if time_steps > con_batch:
            selected_indices = torch.randperm(time_steps)[:con_batch]
            features = features[selected_indices]
            time_steps = con_batch  # Update time_steps to the new size
        # Shape: (num_meta, num_agents * time_steps, feature_dim)
        features = features.permute(1, 2, 0, 3).reshape(num_meta, num_agents * time_steps, feature_dim)
        for meta in range(num_meta):
            # Shape: (num_agents * time_steps, feature_dim)
            meta_features = features[meta]
            # Shuffle and flatten features of each agent. Shape: (num_agents * time_steps, feature_dim)
            shuffled_features = meta_features.reshape(num_agents, time_steps, feature_dim).clone()
            for i in range(num_agents):
                shuffled_indices = torch.randperm(time_steps)
                shuffled_features[i] = meta_features.reshape(num_agents, time_steps, feature_dim)[i, shuffled_indices]
            shuffled_features = shuffled_features.reshape(num_agents * time_steps, feature_dim)

            # Normalize features to unit vectors
            # Shape: (num_agents * time_steps, feature_dim)
            meta_features = F.normalize(meta_features, p=2, dim=1)
            # Shape: (num_agents * time_steps, feature_dim)
            shuffled_features = F.normalize(shuffled_features, p=2, dim=1)

            # Compute cosine similarity for positive and negative pairs
            # Shape: (num_agents * time_steps,)
            positive_similarity = torch.sum(meta_features * shuffled_features, dim=1)
            # Shape: (num_agents * time_steps, num_agents * time_steps)
            negative_similarity = torch.matmul(meta_features, shuffled_features.T)

            # Mask out self-comparisons to avoid self-pairing in negative pairs
            # Shape: (num_agents * time_steps, num_agents * time_steps)
            mask = torch.eye(num_agents * time_steps, device=features.device)
            negative_similarity.masked_fill_(mask.bool(), float('-inf'))

            # Concatenate similarities and calculate the cross-entropy loss
            # Shape: (num_agents * time_steps, 1 + num_agents * time_steps)
            logits = torch.cat([positive_similarity.unsqueeze(1), negative_similarity], dim=1) / con_temp
            # Shape: (num_agents * time_steps,)
            labels = torch.zeros(num_agents * time_steps, dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, labels)
            a_cl_loss += loss

        # Update actor network
        network.actor_optimizer.zero_grad()
        (policy_loss - el_factor * entropy_loss + pl_factor * vae_loss + cl_factor * a_cl_loss).backward()
        if grad_clip is not None:
            actor_norm = torch.nn.utils.clip_grad_norm_(network.actor_network.parameters(), grad_clip)
        else:
            actor_norm = get_gard_norm(network.actor_network.parameters())
        network.actor_optimizer.step()

        clip_frac = clip_mask
        clip_frac = convert_to_tensor(data=clip_frac, data_type=torch.float32, device=device).mean()

        p_l += convert_to_item(policy_loss)
        e_l += convert_to_item(entropy_loss)
        a_gn += convert_to_item(actor_norm)
        c_f += convert_to_item(clip_frac)
        a_v_l += convert_to_item(vae_loss)
        a_c_l += convert_to_item(a_cl_loss)

    for k in range(k_epoch):
        network.critic_optimizer.zero_grad()
        values, _, prediction, mu, logvar = network.forward_v(batch_obs, batch_phase_vec, batch_int_attr_vec, batch_phase_mask, None, batch_neighbor_actions_vec, num_meta)

        # Calculate value loss
        value_loss = torch.square(values.squeeze(-1) - batch_target_values)
        value_loss = torch.mean(value_loss)

        # Calculate prediction loss for Critic VAE
        prediction = prediction.gather(2, batch_actions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, prediction.size(-1))).reshape(prediction.size(0), prediction.size(1), -1)
        mu = mu.gather(2, batch_actions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, mu.size(-1))).reshape(mu.size(0), mu.size(1), -1)
        logvar = logvar.gather(2, batch_actions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, logvar.size(-1))).reshape(logvar.size(0), logvar.size(1), -1)
        vae_loss = torch.mean(torch.square(batch_target_prediction.reshape(prediction.size()) - prediction)) - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Calculate contrastive loss for Critic network
        c_cl_loss = 0
        # Shape: (time_steps, num_meta, n_agents, feature_dim)
        features = mu.reshape(mu.size(0), num_meta, -1, network.critic_network.vae_hidden_dim)
        time_steps, num_meta, num_agents, feature_dim = features.size()
        # Select a random subset of 256 timesteps if there are more than 256 timesteps available
        if time_steps > con_batch:
            selected_indices = torch.randperm(time_steps)[:con_batch]
            features = features[selected_indices]
            time_steps = con_batch  # Update time_steps to the new size
        # Shape: (num_meta, num_agents * time_steps, feature_dim)
        features = features.permute(1, 2, 0, 3).reshape(num_meta, num_agents * time_steps, feature_dim)
        for meta in range(num_meta):
            # Shape: (num_agents * time_steps, feature_dim)
            meta_features = features[meta]
            # Shuffle and flatten features of each agent. Shape: (num_agents * time_steps, feature_dim)
            shuffled_features = meta_features.reshape(num_agents, time_steps, feature_dim).clone()
            for i in range(num_agents):
                shuffled_indices = torch.randperm(time_steps)
                shuffled_features[i] = meta_features.reshape(num_agents, time_steps, feature_dim)[i, shuffled_indices]
            shuffled_features = shuffled_features.reshape(num_agents * time_steps, feature_dim)

            # Normalize features to unit vectors
            # Shape: (num_agents * time_steps, feature_dim)
            meta_features = F.normalize(meta_features, p=2, dim=1)
            # Shape: (num_agents * time_steps, feature_dim)
            shuffled_features = F.normalize(shuffled_features, p=2, dim=1)

            # Compute cosine similarity for positive and negative pairs
            # Shape: (num_agents * time_steps,)
            positive_similarity = torch.sum(meta_features * shuffled_features, dim=1)
            # Shape: (num_agents * time_steps, num_agents * time_steps)
            negative_similarity = torch.matmul(meta_features, shuffled_features.T)

            # Mask out self-comparisons to avoid self-pairing in negative pairs
            # Shape: (num_agents * time_steps, num_agents * time_steps)
            mask = torch.eye(num_agents * time_steps, device=features.device)
            negative_similarity.masked_fill_(mask.bool(), float('-inf'))

            # Concatenate similarities and calculate the cross-entropy loss
            # Shape: (num_agents * time_steps, 1 + num_agents * time_steps)
            logits = torch.cat([positive_similarity.unsqueeze(1), negative_similarity], dim=1) / con_temp
            # Shape: (num_agents * time_steps,)
            labels = torch.zeros(num_agents * time_steps, dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, labels)
            c_cl_loss += loss

        # Backward
        network.critic_optimizer.zero_grad()
        (vl_factor * value_loss + pl_factor * vae_loss + cl_factor * c_cl_loss).backward()

        # Update critic network
        if grad_clip is not None:
            critic_norm = torch.nn.utils.clip_grad_norm_(network.critic_network.parameters(), grad_clip)
        else:
            critic_norm = get_gard_norm(network.critic_network.parameters())

        network.critic_optimizer.step()

        v_l += convert_to_item(value_loss)
        c_gn += convert_to_item(critic_norm)
        c_v_l += convert_to_item(vae_loss)
        c_c_l += convert_to_item(c_cl_loss)

    return (p_l / k_epoch, v_l / k_epoch, e_l / k_epoch, a_gn / k_epoch, c_gn / k_epoch,
            c_f / k_epoch, a_v_l / k_epoch, c_v_l / k_epoch, a_c_l / k_epoch, c_c_l / k_epoch)



def main():
    from training import entrypoint
    entrypoint('Unicorn', Unicorn, Runner, get_global_train_buffer,
               calculate_gradients_ma_ppo, write_to_Tensorboard)


if __name__ == '__main__':
    main()
