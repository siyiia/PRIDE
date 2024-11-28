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

device = 'cuda'


# def find_max(r_t_single):
#     window = 3
#     max_sum = -np.inf
#     max_index = -1
#
#     for i in range(1, r_t_single.shape[0] + 1):
#         window_sum = np.sum(r_t_single[max(0, i - window):i])
#         if window_sum > max_sum:
#             max_sum = window_sum
#             max_index = i - 2
#
#     return max_index, max_sum
#
#
# def find_min(r_t_single):
#     window = 3
#     min_sum = np.inf
#     min_index = -1
#
#     for i in range(1, r_t_single.shape[0] + 1):
#         window_sum = np.sum(r_t_single[max(0, i - window):i])
#         if window_sum < min_sum:
#             min_sum = window_sum
#             min_index = i - 2
#
#     return min_index, min_sum

def find_max(r_t_single, window):
    max_sum = -np.inf
    max_index = -1

    for i in range(1, r_t_single.shape[0] + 1):
        window_sum = np.sum(r_t_single[max(0, i - window):i])
        if window_sum > max_sum:
            max_sum = window_sum
            max_index = i - 1

    return max_index, max_sum


def find_min(r_t_single, window):
    min_sum = np.inf
    min_index = -1

    for i in range(window, r_t_single.shape[0] + 1):
        window_sum = np.sum(r_t_single[max(0, i - window):i])
        if window_sum < min_sum:
            min_sum = window_sum
            min_index = i - 1

    return min_index, min_sum


class RewardHighlightModel(RewardModel):
    def __init__(self, ds, da,
                 ensemble_size=3, lr=3e-4, mb_size = 128, size_segment=1,
                 env_maker=None, max_size=100, activation='tanh', capacity=5e5,
                 large_batch=1, label_margin=0.0,
                 teacher_beta=-1, teacher_gamma=1,
                 teacher_eps_mistake=0,
                 teacher_eps_skip=0,
                 teacher_eps_equal=0,
                 positive_weight=0.1,
                 negative_weight=0.1,
                 smallest_rew_threshold=0,
                 largest_rew_threshold=0,
                 window=5
                 ):
        super().__init__(ds, da, ensemble_size, lr, mb_size, size_segment,
                         env_maker, max_size, activation, capacity,
                         large_batch, label_margin,
                         teacher_beta, teacher_gamma,
                         teacher_eps_mistake, teacher_eps_skip,
                         teacher_eps_equal)

        self.buffer_critical_point = np.empty((self.capacity, 2), dtype=np.int)

        self.pos_discount_start_multiplier = 1.0
        self.pos_discount = 0.7
        self.min_pos_discount = 0.01

        self.neg_discount_start_multiplier = 1.0
        self.neg_discount = 0.7
        self.min_neg_discount = 0.01

        self.positive_weight = positive_weight
        self.negative_weight = negative_weight

        self.smallest_rew_threshold = smallest_rew_threshold
        self.largest_rew_threshold = largest_rew_threshold

        self.window = window

    def put_queries(self, sa_t_1, sa_t_2, labels, critical_points):
        total_sample = sa_t_1.shape[0]
        next_index = self.buffer_index + total_sample
        if next_index >= self.capacity:
            self.buffer_full = True
            maximum_index = self.capacity - self.buffer_index
            np.copyto(self.buffer_seg1[self.buffer_index:self.capacity], sa_t_1[:maximum_index])
            np.copyto(self.buffer_seg2[self.buffer_index:self.capacity], sa_t_2[:maximum_index])
            np.copyto(self.buffer_label[self.buffer_index:self.capacity], labels[:maximum_index])
            np.copyto(self.buffer_critical_point[self.buffer_index:self.capacity], critical_points[:maximum_index])

            remain = total_sample - (maximum_index)
            if remain > 0:
                np.copyto(self.buffer_seg1[0:remain], sa_t_1[maximum_index:])
                np.copyto(self.buffer_seg2[0:remain], sa_t_2[maximum_index:])
                np.copyto(self.buffer_label[0:remain], labels[maximum_index:])
                np.copyto(self.buffer_critical_point[0:remain], critical_points[maximum_index:])

            self.buffer_index = remain
        else:
            np.copyto(self.buffer_seg1[self.buffer_index:next_index], sa_t_1)
            np.copyto(self.buffer_seg2[self.buffer_index:next_index], sa_t_2)
            np.copyto(self.buffer_label[self.buffer_index:next_index], labels)
            np.copyto(self.buffer_critical_point[self.buffer_index:next_index], critical_points)
            self.buffer_index = next_index


    def find_critical_points(self, r_t_1, r_t_2, label):
        critical_points = []

        for i in range(r_t_1.shape[0]):
            largest_rew_index = -1
            smallest_rew_index = -1
            # 1 max 2 min
            if label[i] == 0:
                temp_max_index, temp_max_sum = find_max(r_t_1[i], 5)
                temp_min_index, temp_min_sum = find_min(r_t_2[i], 5)
            # 1 min 2 max
            elif label[i] == 1:
                temp_max_index, temp_max_sum = find_max(r_t_2[i], 5)
                temp_min_index, temp_min_sum = find_min(r_t_1[i], 5)

            # if temp_max_sum > self.largest_rew_threshold:
            #     largest_rew_index = temp_max_index
            # if temp_min_sum < self.smallest_rew_threshold:
            #     smallest_rew_index = temp_min_index

            data = [temp_min_sum, temp_max_sum]
            data_np = np.array(data, dtype=np.float32)
            # data_tensor = torch.tensor(data, dtype=torch.float32)
            # prob = F.softmax(data_tensor, dim=-1)
            prob = data_np/np.sum(data_np)

            if temp_min_sum < self.smallest_rew_threshold:
                smallest_rew_index = temp_min_index
                if temp_max_sum > self.largest_rew_threshold:
                    largest_rew_index = temp_max_index

            elif prob[1] > 0.9:
                largest_rew_index = temp_max_index
                smallest_rew_index = temp_min_index

            critical_points.append([smallest_rew_index, largest_rew_index])

        return np.asarray(critical_points)



    def get_label(self, sa_t_1, sa_t_2, r_t_1, r_t_2):
        sum_r_t_1 = np.sum(r_t_1, axis=1)
        sum_r_t_2 = np.sum(r_t_2, axis=1)

        # skip the query
        if self.teacher_thres_skip > 0:
            max_r_t = np.maximum(sum_r_t_1, sum_r_t_2)
            max_index = (max_r_t > self.teacher_thres_skip).reshape(-1)
            if sum(max_index) == 0:
                return None, None, None, None, []

            sa_t_1 = sa_t_1[max_index]
            sa_t_2 = sa_t_2[max_index]
            r_t_1 = r_t_1[max_index]
            r_t_2 = r_t_2[max_index]
            sum_r_t_1 = np.sum(r_t_1, axis=1)
            sum_r_t_2 = np.sum(r_t_2, axis=1)

        # equally preferable
        margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) < self.teacher_thres_equal).reshape(-1)

        # perfectly rational
        seg_size = r_t_1.shape[1]
        temp_r_t_1 = r_t_1.copy()
        temp_r_t_2 = r_t_2.copy()
        for index in range(seg_size - 1):
            temp_r_t_1[:, :index + 1] *= self.teacher_gamma
            temp_r_t_2[:, :index + 1] *= self.teacher_gamma
        sum_r_t_1 = np.sum(temp_r_t_1, axis=1)
        sum_r_t_2 = np.sum(temp_r_t_2, axis=1)

        rational_labels = 1 * (sum_r_t_1 < sum_r_t_2)
        if self.teacher_beta > 0:  # Bradley-Terry rational model
            r_hat = torch.cat([torch.Tensor(sum_r_t_1),
                               torch.Tensor(sum_r_t_2)], axis=-1)
            r_hat = r_hat * self.teacher_beta
            ent = F.softmax(r_hat, dim=-1)[:, 1]
            labels = torch.bernoulli(ent).int().numpy().reshape(-1, 1)
        else:
            labels = rational_labels

        # making a mistake
        len_labels = labels.shape[0]
        rand_num = np.random.rand(len_labels)
        noise_index = rand_num <= self.teacher_eps_mistake
        labels[noise_index] = 1 - labels[noise_index]

        # equally preferable
        labels[margin_index] = -1

        #################################################
        critical_r_t_1 = r_t_1.copy()
        critical_r_t_2 = r_t_2.copy()
        critical_points = self.find_critical_points(critical_r_t_1, critical_r_t_2, labels)

        return sa_t_1, sa_t_2, r_t_1, r_t_2, labels, critical_points

    def kcenter_sampling(self):

        # get queries
        num_init = self.mb_size * self.large_batch
        sa_t_1, sa_t_2, r_t_1, r_t_2 = self.get_queries(
            mb_size=num_init)

        # get final queries based on kmeans clustering
        temp_sa_t_1 = sa_t_1[:, :, :self.ds]
        temp_sa_t_2 = sa_t_2[:, :, :self.ds]
        temp_sa = np.concatenate([temp_sa_t_1.reshape(num_init, -1),
                                  temp_sa_t_2.reshape(num_init, -1)], axis=1)

        max_len = self.capacity if self.buffer_full else self.buffer_index

        tot_sa_1 = self.buffer_seg1[:max_len, :, :self.ds]
        tot_sa_2 = self.buffer_seg2[:max_len, :, :self.ds]
        tot_sa = np.concatenate([tot_sa_1.reshape(max_len, -1),
                                 tot_sa_2.reshape(max_len, -1)], axis=1)

        selected_index = KCenterGreedy(temp_sa, tot_sa, self.mb_size)

        r_t_1, sa_t_1 = r_t_1[selected_index], sa_t_1[selected_index]
        r_t_2, sa_t_2 = r_t_2[selected_index], sa_t_2[selected_index]

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels, critical_points = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)

        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels, critical_points)

        return len(labels)

    def kcenter_disagree_sampling(self):

        num_init = self.mb_size * self.large_batch
        num_init_half = int(num_init * 0.5)

        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 = self.get_queries(
            mb_size=num_init)

        # get final queries based on uncertainty
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[:num_init_half]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]

        # get final queries based on kmeans clustering
        temp_sa_t_1 = sa_t_1[:, :, :self.ds]
        temp_sa_t_2 = sa_t_2[:, :, :self.ds]

        temp_sa = np.concatenate([temp_sa_t_1.reshape(num_init_half, -1),
                                  temp_sa_t_2.reshape(num_init_half, -1)], axis=1)

        max_len = self.capacity if self.buffer_full else self.buffer_index

        tot_sa_1 = self.buffer_seg1[:max_len, :, :self.ds]
        tot_sa_2 = self.buffer_seg2[:max_len, :, :self.ds]
        tot_sa = np.concatenate([tot_sa_1.reshape(max_len, -1),
                                 tot_sa_2.reshape(max_len, -1)], axis=1)

        selected_index = KCenterGreedy(temp_sa, tot_sa, self.mb_size)

        r_t_1, sa_t_1 = r_t_1[selected_index], sa_t_1[selected_index]
        r_t_2, sa_t_2 = r_t_2[selected_index], sa_t_2[selected_index]

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels, critical_points = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)

        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels, critical_points)

        return len(labels)

    def kcenter_entropy_sampling(self):

        num_init = self.mb_size * self.large_batch
        num_init_half = int(num_init * 0.5)

        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 = self.get_queries(
            mb_size=num_init)

        # get final queries based on uncertainty
        entropy, _ = self.get_entropy(sa_t_1, sa_t_2)
        top_k_index = (-entropy).argsort()[:num_init_half]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]

        # get final queries based on kmeans clustering
        temp_sa_t_1 = sa_t_1[:, :, :self.ds]
        temp_sa_t_2 = sa_t_2[:, :, :self.ds]

        temp_sa = np.concatenate([temp_sa_t_1.reshape(num_init_half, -1),
                                  temp_sa_t_2.reshape(num_init_half, -1)], axis=1)

        max_len = self.capacity if self.buffer_full else self.buffer_index

        tot_sa_1 = self.buffer_seg1[:max_len, :, :self.ds]
        tot_sa_2 = self.buffer_seg2[:max_len, :, :self.ds]
        tot_sa = np.concatenate([tot_sa_1.reshape(max_len, -1),
                                 tot_sa_2.reshape(max_len, -1)], axis=1)

        selected_index = KCenterGreedy(temp_sa, tot_sa, self.mb_size)

        r_t_1, sa_t_1 = r_t_1[selected_index], sa_t_1[selected_index]
        r_t_2, sa_t_2 = r_t_2[selected_index], sa_t_2[selected_index]

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels, critical_points = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)

        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels, critical_points)

        return len(labels)

    def uniform_sampling(self):
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 = self.get_queries(
            mb_size=self.mb_size)

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels, critical_points = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)

        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels, critical_points)

        return len(labels)

    def disagreement_sampling(self):

        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 = self.get_queries(
            mb_size=self.mb_size * self.large_batch)

        # get final queries based on uncertainty
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[:self.mb_size]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels, critical_points = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels, critical_points)

        return len(labels)

    def entropy_sampling(self):

        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 = self.get_queries(
            mb_size=self.mb_size * self.large_batch)

        # get final queries based on uncertainty
        entropy, _ = self.get_entropy(sa_t_1, sa_t_2)

        top_k_index = (-entropy).argsort()[:self.mb_size]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels, critical_points = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)

        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels, critical_points)

        return len(labels)


    def generate_critical_point_segment(self, critical_points):
        rolled_critical_points = [[[0, 0] for _ in range(self.size_segment)] for _ in range(len(critical_points))]
        for i in range(len(critical_points)):
            neg_index = critical_points[i][0]
            pos_index = critical_points[i][1]

            if pos_index != -1:
                current_pos_discount = self.pos_discount_start_multiplier
                for j in reversed(range(max(0, pos_index - self.window), pos_index + 1)):
                    rolled_critical_points[i][j][1] = max(current_pos_discount, self.min_pos_discount)
                    current_pos_discount *= self.pos_discount

            if neg_index != -1:
                current_neg_discount = self.neg_discount_start_multiplier
                for j in reversed(range(max(0, neg_index - self.window), neg_index + 1)):
                    rolled_critical_points[i][j][0] = max(current_neg_discount, self.min_neg_discount)
                    current_neg_discount *= self.neg_discount
        critical_points = np.asarray(rolled_critical_points).astype('float32')
        critical_points = torch.tensor(critical_points).to(device)
        return critical_points

    def get_critical_points_rewards(self, critical_points, labels, r_hat1_ind, r_hat2_ind):
        critical_points = self.generate_critical_point_segment(critical_points)

        critical_points_discounted_reward_punishment = torch.zeros_like(r_hat1_ind)
        critical_points_discounted_reward_approve = torch.zeros_like(r_hat1_ind)
        for i in range(len(labels)):
            if labels[i] == 0:
                critical_points_discounted_reward_punishment[i] = r_hat2_ind[i] * critical_points[i, :, 0].unsqueeze(1)
                critical_points_discounted_reward_approve[i] = r_hat1_ind[i] * critical_points[i, :, 1].unsqueeze(1)
            if labels[i] == 1:
                critical_points_discounted_reward_punishment[i] = r_hat1_ind[i] * critical_points[i, :, 0].unsqueeze(1)
                critical_points_discounted_reward_approve[i] = r_hat2_ind[i] * critical_points[i, :, 1].unsqueeze(1)

        n_approve = torch.sum(critical_points[:, :, 1] == 1).item()
        n_punishment = torch.sum(critical_points[:, :, 0] == 1).item()

        punishment_reward = torch.sum(critical_points_discounted_reward_punishment)  # / punishments_in_batch
        approve_reward = torch.sum(critical_points_discounted_reward_approve)  # / approvements_in_batch


        return approve_reward, punishment_reward, n_approve, n_punishment

    def train_reward(self):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])

        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))

        num_epochs = int(np.ceil(max_len/self.train_batch_size))
        list_debug_loss1, list_debug_loss2 = [], []
        total = 0

        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss = 0.0

            last_index = (epoch+1)*self.train_batch_size
            if last_index > max_len:
                last_index = max_len

            for member in range(self.de):

                # get random batch
                idxs = total_batch_index[member][epoch*self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs]
                sa_t_2 = self.buffer_seg2[idxs]
                labels = self.buffer_label[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(device)
                critical_points = self.buffer_critical_point[idxs]

                if member == 0:
                    total += labels.size(0)

                # get logits
                r_hat1_ind = self.r_hat_member(sa_t_1, member=member)
                r_hat2_ind = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1_ind.sum(axis=1)
                r_hat2 = r_hat2_ind.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                ################################################
                approve_reward, punishment_reward, n_approve, n_punishment = self.get_critical_points_rewards(
                    critical_points, labels, r_hat1_ind, r_hat2_ind)
                n_approve = n_approve if n_approve != 0 else 1
                n_punishment = n_punishment if n_punishment != 0 else 1

                approve_reward = approve_reward / n_approve
                punishment_reward = punishment_reward / n_punishment


                # compute loss
                curr_loss = self.CEloss(r_hat, labels) - approve_reward * self.positive_weight + punishment_reward * self.negative_weight
                # curr_loss = (1 - self.positive_weight - self.negative_weight) * self.CEloss(r_hat, labels) - approve_reward * self.positive_weight + punishment_reward * self.negative_weight
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())

                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct

            loss.backward()
            self.opt.step()

        ensemble_acc = ensemble_acc / total

        return ensemble_acc

    def train_soft_reward(self):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])

        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))

        num_epochs = int(np.ceil(max_len/self.train_batch_size))
        list_debug_loss1, list_debug_loss2 = [], []
        total = 0

        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss = 0.0

            last_index = (epoch+1)*self.train_batch_size
            if last_index > max_len:
                last_index = max_len

            for member in range(self.de):

                # get random batch
                idxs = total_batch_index[member][epoch*self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs]
                sa_t_2 = self.buffer_seg2[idxs]
                labels = self.buffer_label[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(device)
                critical_points = self.buffer_critical_point[idxs]

                if member == 0:
                    total += labels.size(0)

                # get logits
                r_hat1_ind = self.r_hat_member(sa_t_1, member=member)
                r_hat2_ind = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1_ind.sum(axis=1)
                r_hat2 = r_hat2_ind.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                ################################################
                approve_reward, punishment_reward, n_approve, n_punishment = self.get_critical_points_rewards(
                    critical_points, labels, r_hat1_ind, r_hat2_ind)
                n_approve = n_approve if n_approve != 0 else 1
                n_punishment = n_punishment if n_punishment != 0 else 1

                approve_reward = approve_reward / n_approve
                punishment_reward = punishment_reward / n_punishment

                # compute loss
                uniform_index = labels == -1
                labels[uniform_index] = 0
                target_onehot = torch.zeros_like(r_hat).scatter(1, labels.unsqueeze(1), self.label_target)
                target_onehot += self.label_margin
                if sum(uniform_index) > 0:
                    target_onehot[uniform_index] = 0.5
                curr_loss = self.softXEnt_loss(r_hat, target_onehot) - approve_reward * self.positive_weight + punishment_reward * self.negative_weight
                # curr_loss = (1 - self.positive_weight - self.negative_weight) * self.softXEnt_loss(r_hat, target_onehot) - approve_reward * self.positive_weight + punishment_reward * self.negative_weight
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())

                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct

            loss.backward()
            self.opt.step()

        ensemble_acc = ensemble_acc / total

        return ensemble_acc