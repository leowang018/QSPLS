import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import itertools
import tqdm
import copy
import scipy.stats as st
import os
import time

from scipy.stats import norm
from reward_model import RewardModel
from VaeModel import VAE
from sklearn.cluster import KMeans


class RewardModelQSPLS(RewardModel):
    def __init__(self, ds, da, device, ensemble_size=3, lr=3e-4, mb_size=128, size_segment=1,
                 env_maker=None, max_size=100, activation='tanh', capacity=5e5,
                 large_batch=1, label_margin=0.0,
                 teacher_beta=-1, teacher_gamma=1,
                 teacher_eps_mistake=0,
                 teacher_eps_skip=0,
                 teacher_eps_equal=0,
                 inv_label_ratio=10,
                 mu=1,
                 weight_factor=1.0,
                 adv_mu=2,
                 path=None,
                 data_aug_ratio=1,
                 k_means=10,
                 kl_lamda=0.5,
                 similarity_threshold=0.5
                 ):

        self.ds = ds
        self.da = da
        self.device = device
        self.de = ensemble_size
        self.lr = lr
        self.ensemble = []
        self.paramlst = []
        self.opt = None
        self.model = None
        self.max_size = max_size
        self.activation = activation
        self.size_segment = size_segment
        self.path = path
        self.data_aug_ratio = data_aug_ratio
        self.count = 0

        self.capacity = int(capacity)
        self.buffer_seg1 = np.empty((self.capacity, size_segment, self.ds + self.da), dtype=np.float32)
        self.buffer_seg2 = np.empty((self.capacity, size_segment, self.ds + self.da), dtype=np.float32)

        self.buffer_label = np.empty((self.capacity, 1), dtype=np.float32)
        self.buffer_mask = np.ones((self.capacity, 1), dtype=np.float32)
        self.buffer_index = 0
        self.buffer_full = False

        self.construct_ensemble()
        self.vae_model = None
        self.vae_opt = None
        self.construct_vae_model()
        self.inputs = []
        self.targets = []
        self.raw_actions = []
        self.img_inputs = []
        self.mb_size = mb_size
        self.origin_mb_size = mb_size
        self.train_batch_size = 128
        self.CEloss = nn.CrossEntropyLoss()
        self.CEloss_ = nn.CrossEntropyLoss(reduction='none')
        self.running_means = []
        self.running_stds = []
        self.best_seg = []
        self.best_label = []
        self.best_action = []
        self.large_batch = large_batch
        self.mu = mu
        self.weight_factor = weight_factor
        self.adv_mu = adv_mu
        self.obs_l = 0
        self.action_l = 0

        # new teacher
        self.teacher_beta = teacher_beta
        self.teacher_gamma = teacher_gamma
        self.teacher_eps_mistake = teacher_eps_mistake
        self.teacher_eps_equal = teacher_eps_equal
        self.teacher_eps_skip = teacher_eps_skip
        self.teacher_thres_skip = 0
        self.teacher_thres_equal = 0

        self.label_margin = label_margin
        self.label_target = 1 - 2 * self.label_margin

        self.inv_label_ratio = inv_label_ratio
        self.k_means = k_means
        self.kl_lamda = kl_lamda
        self.similarity_threshold = similarity_threshold

    def construct_vae_model(self):
        self.vae_model = VAE(self.size_segment * self.ds, 32).to(self.device)
        self.vae_opt = torch.optim.Adam(self.vae_model.parameters(), lr=1e-4)

    def qspls_sampling(self):
        sa_t_1, sa_t_2, r_t_1, r_t_2 = self.get_queries(
            mb_size=self.mb_size * self.large_batch)
        sa_t = np.concatenate([sa_t_1, sa_t_2], axis=0)
        r_t = np.concatenate([r_t_1, r_t_2], axis=0)
        # reward-guided adjacent cluster sampling
        cluster_result = self.kmeans_clustering(sa_t, self.vae_model, self.k_means, self.device)
        # form initial query dataset
        traj1_idx, traj2_idx = self.sample_trajectory(cluster_result, self.mb_size * self.large_batch)
        sa_t_1 = sa_t[traj1_idx]
        sa_t_2 = sa_t[traj2_idx]
        r_t_1 = r_t[traj1_idx]
        r_t_2 = r_t[traj2_idx]

        # entropy-based similarity filtering
        filter_index = self.select_high_entropy_diverse_samples(sa_t_1, sa_t_2)[:self.mb_size]

        r_t_1, sa_t_1 = r_t_1[filter_index], sa_t_1[filter_index]
        r_t_2, sa_t_2 = r_t_2[filter_index], sa_t_2[filter_index]

        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)

        return len(labels)

    def select_high_entropy_diverse_samples(self, sa_t_1, sa_t_2):
        entropy_value = self.get_sample_entropy(sa_t_1, sa_t_2)
        dist_matrix = self.cosine_similarity(sa_t_1, sa_t_2)
        top_k_index = (-entropy_value).argsort()
        filtered_indices = [i for i in top_k_index if dist_matrix[i] < self.similarity_threshold]
        return filtered_indices

    def get_sample_entropy(self, x_1, x_2):
        with torch.no_grad():
            r_hat1 = self.r_hat_member(x_1, member=0)
            r_hat2 = self.r_hat_member(x_2, member=0)
            r_hat1 = r_hat1.sum(axis=1)
            r_hat2 = r_hat2.sum(axis=1)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

        ent = F.softmax(r_hat, dim=-1) * F.log_softmax(r_hat, dim=-1)
        entropy_value = ent.sum(axis=-1).abs()
        return entropy_value

    def cosine_similarity(self, x_1, x_2):
        with torch.no_grad():
            latent_sa_1 = self.get_latent_represention(x_1)
            latent_sa_2 = self.get_latent_represention(x_2)
            cos_sim_batch = F.cosine_similarity(latent_sa_1, latent_sa_2, dim=1)
            dist_matrix = cos_sim_batch.cpu().numpy()
        return dist_matrix

    def get_latent_represention(self, x_1):
        state_sequences = x_1[:, :, :self.ds]
        state_sequences = torch.from_numpy(state_sequences).float().to(self.device)
        state_sequences = state_sequences.reshape(state_sequences.shape[0], -1)
        return self.vae_model.encode(state_sequences)

    def get_state_sequence(self, mb_size=20):
        len_traj, max_len = len(self.inputs[0]), len(self.inputs)

        if len(self.inputs[-1]) < len_traj:
            max_len = max_len - 1

        # get train traj
        train_inputs = np.array(self.inputs[:max_len])
        train_targets = np.array(self.targets[:max_len])

        batch_index_1 = np.random.choice(max_len, size=mb_size, replace=True)
        sa_t_1 = train_inputs[batch_index_1]  # Batch x T x dim of s&a
        r_t_1 = train_targets[batch_index_1]  # Batch x T x 1

        sa_t_1 = sa_t_1.reshape(-1, sa_t_1.shape[-1])  # (Batch x T) x dim of s&a
        r_t_1 = r_t_1.reshape(-1, r_t_1.shape[-1])  # (Batch x T) x 1

        # Generate time index
        time_index = np.array([list(range(i * len_traj,
                                          i * len_traj + self.size_segment)) for i in range(mb_size)])
        time_index_1 = time_index + np.random.choice(len_traj - self.size_segment, size=mb_size, replace=True).reshape(
            -1, 1)

        sa_t_1 = np.take(sa_t_1, time_index_1, axis=0)  # Batch x size_seg x dim of s&a
        r_t_1 = np.take(r_t_1, time_index_1, axis=0)
        state_sequence = sa_t_1[:, :, :self.ds]

        return state_sequence, sa_t_1, r_t_1

    def kmeans_clustering(self, sa_t, vae, k, device):
        state_sequences = sa_t[:, :, :self.ds]
        r_hat = self.r_hat_member(sa_t, member=0)
        r_hat_sum = r_hat.sum(axis=1).detach().cpu().numpy()
        with torch.no_grad():
            state_sequences = torch.from_numpy(state_sequences).float().to(device)
            state_sequences = state_sequences.reshape(state_sequences.shape[0], -1)
            latent_representations = vae.encode(state_sequences)

        latent_representations_np = latent_representations.cpu().numpy()

        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        cluster_labels = kmeans.fit_predict(latent_representations_np)

        cluster_rewards = np.zeros(k)
        cluster_index_dict = {}
        for i in range(k):
            cluster_indices = np.where(cluster_labels == i)[0]
            cluster_index_dict[i] = cluster_indices
            cluster_rewards[i] = np.mean(r_hat_sum[cluster_indices])

        sorted_indices = np.argsort(cluster_rewards)
        sorted_cluster_index_dict = {}
        for new_label, old_label in enumerate(sorted_indices):
            sorted_cluster_index_dict[new_label] = cluster_index_dict[old_label]

        return sorted_cluster_index_dict

    def sample_trajectory(self, cluster_label, mb_size):
        available_clusters = [cluster for cluster in cluster_label if len(cluster_label[cluster]) > 1]
        trajectory_1_index = []
        trajectory_2_index = []
        for _ in range(mb_size):
            cluster_1 = np.random.randint(0, len(available_clusters) - 1)
            traj_1_index = np.random.choice(cluster_label[cluster_1])
            traj_2_index = np.random.choice(cluster_label[cluster_1 + 1])
            trajectory_1_index.append(traj_1_index)
            trajectory_2_index.append(traj_2_index)

        return trajectory_1_index, trajectory_2_index





























