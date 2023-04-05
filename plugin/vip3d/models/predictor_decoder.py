import random
from typing import Dict, List, Tuple, NamedTuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .predictor_lib import PointSubGraph, GlobalGraphRes, CrossAttention, GlobalGraph, MLP
from .. import utils as utils


class DecoderRes(nn.Module):
    def __init__(self, hidden_size, out_features=60):
        super(DecoderRes, self).__init__()
        self.mlp = MLP(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, out_features)

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.mlp(hidden_states)
        hidden_states = self.fc(hidden_states)
        return hidden_states


class DecoderResCat(nn.Module):
    def __init__(self, hidden_size, in_features, out_features=60):
        super(DecoderResCat, self).__init__()
        self.mlp = MLP(in_features, hidden_size)
        self.fc = nn.Linear(hidden_size + in_features, out_features)

    def forward(self, hidden_states, **kwargs):
        hidden_states = torch.cat([hidden_states, self.mlp(hidden_states)], dim=-1)
        hidden_states = self.fc(hidden_states)
        return hidden_states


class Decoder(nn.Module):

    def __init__(self,
                 vectornet,
                 variety_loss=False,
                 variety_loss_prob=False,
                 nms_threshold=3.0,
                 goals_2D=False,
                 hidden_size=128,
                 future_frame_num=12,
                 mode_num=6,
                 do_eval=False,
                 train_pred_probs_only=False,
                 K_is_1=False,
                 K_is_1_constant=False,
                 exclude_static=False,
                 reduce_prob_of=None,
                 rebalance_prob=False,
                 ):
        super(Decoder, self).__init__()

        self.variety_loss = variety_loss
        self.variety_loss_prob = variety_loss_prob

        self.nms_threshold = nms_threshold
        self.goals_2D = goals_2D
        self.future_frame_num = future_frame_num
        self.mode_num = mode_num
        self.do_eval = do_eval
        self.train_pred_probs_only = train_pred_probs_only
        self.train_pred_probs_only_values = 270.0 / np.array([1600, 340, 270, 800, 1680, 12000], dtype=float)

        self.decoder = DecoderRes(hidden_size, out_features=2)

        if self.variety_loss:
            self.variety_loss_decoder = DecoderResCat(hidden_size, hidden_size, out_features=6 * self.future_frame_num * 2)

            if variety_loss_prob:
                if self.train_pred_probs_only:
                    self.train_pred_probs_only_decocer = DecoderResCat(hidden_size, hidden_size, out_features=6)
                self.variety_loss_decoder = DecoderResCat(hidden_size, hidden_size, out_features=6 * self.future_frame_num * 2 + 6)
        else:
            assert False

        self.K_is_1 = K_is_1
        self.K_is_1_constant = K_is_1_constant
        self.exclude_static = exclude_static
        self.reduce_prob_of = reduce_prob_of
        self.rebalance_prob = rebalance_prob

    def forward_variety_loss(self, mapping: List[Dict], hidden_states: Tensor, batch_size, inputs: Tensor,
                             inputs_lengths: List[int], labels_is_valid: List[np.ndarray], loss: Tensor,
                             DE: np.ndarray, device, labels: List[np.ndarray],
                             agents_indices=None,
                             agents=None):
        """
        :param hidden_states: hidden states of all elements after encoding by global graph (shape [batch_size, -1, hidden_size])
        :param inputs: hidden states of all elements before encoding by global graph (shape [batch_size, 'element num', hidden_size])
        :param inputs_lengths: valid element number of each example
        :param DE: displacement error (shape [batch_size, self.future_frame_num])
        """
        assert batch_size == 1

        agent_num = len(agents_indices) if agents_indices is not None else agents[0].shape[0]
        assert agent_num == labels[0].shape[0]
        if True:
            if agents_indices is not None:
                agents_indices = torch.tensor(agents_indices, dtype=torch.long, device=device)
                hidden_states = hidden_states[:, agents_indices, :]
            else:
                hidden_states = hidden_states[:, :agent_num, :]

            outputs = self.variety_loss_decoder(hidden_states)

            if True:
                if self.train_pred_probs_only:
                    pred_probs = F.log_softmax(self.train_pred_probs_only_decocer(hidden_states), dim=-1)
                else:
                    pred_probs = F.log_softmax(outputs[:, :, -6:], dim=-1)
                outputs = outputs[:, :, :-6].view([batch_size, agent_num, 6, self.future_frame_num, 2])
            else:
                outputs = outputs.view([batch_size, agent_num, 6, self.future_frame_num, 2])

        for i in range(batch_size):
            if True:
                if self.rebalance_prob:
                    if not hasattr(self, 'rebalance_prob_values'):
                        self.rebalance_prob_values = np.zeros(6, dtype=int)

                valid_agent_num = 0
                for agent_idx in range(agent_num):
                    should_train = True

                    gt_points = np.array(labels[i][agent_idx]).reshape([self.future_frame_num, 2])
                    last_valid_index = -1
                    for j in range(self.future_frame_num)[::-1]:
                        if labels_is_valid[i][agent_idx, j]:
                            last_valid_index = j
                            break
                    if self.K_is_1:
                        argmin = 0
                        if self.K_is_1_constant:
                            past_boxes_list = mapping[i]['past_boxes_list']
                            assert len(past_boxes_list) == agent_num
                            past_boxes = past_boxes_list[agent_idx]
                            if utils.get_dis_point2point(past_boxes[1, :2]) > utils.eps:
                                dis = utils.get_dis_point2point(past_boxes[1, :2], past_boxes[2, :2])
                                for k in range(12):
                                    outputs[i, agent_idx, 0, k, 0] = dis * (k + 1)
                                    outputs[i, agent_idx, 0, k, 1] = 0.

                                # outputs[i, agent_idx, 0] = torch.tensor([], , device=device)

                    else:
                        argmin = np.argmin(utils.get_dis_point_2_points(gt_points[last_valid_index], utils.to_numpy(outputs[i, agent_idx, :, last_valid_index, :])))

                    # argmin = utils.argmin_traj(labels[i][agent_idx], labels_is_valid[i][agent_idx], utils.to_numpy(outputs[i, agent_idx]))
                    loss_ = F.smooth_l1_loss(outputs[i, agent_idx, argmin],
                                             torch.tensor(labels[i][agent_idx], device=device, dtype=torch.float), reduction='none')

                    if self.rebalance_prob:
                        self.rebalance_prob_values[argmin] += 1

                    if self.train_pred_probs_only:
                        if np.random.rand() > self.train_pred_probs_only_values[argmin]:
                            should_train = False

                        # if np.random.randint(0, 50) == 0:
                        #     torch.set_printoptions(precision=3, sci_mode=False)
                        #     print('pred_probs[i, agent_idx]', pred_probs[i, agent_idx].exp())
                        #     print(outputs[i, agent_idx, :, -1])

                    if self.reduce_prob_of is not None:
                        if np.random.randint(0, 50) == 0:
                            torch.set_printoptions(precision=3, sci_mode=False)
                            print('pred_probs[i, agent_idx]', pred_probs[i, agent_idx].exp())
                            print(outputs[i, agent_idx, :, -1])
                        pred_probs[i, agent_idx] = pred_probs[i, agent_idx].exp()
                        pred_probs[i, agent_idx, 5] *= 0.1
                        pred_probs[i, agent_idx, 1] *= 3.0
                        pred_probs[i, agent_idx, 2] *= 3.0
                        pred_probs[i, agent_idx] = pred_probs[i, agent_idx].log()
                        # past_boxes_list = mapping[0]['past_boxes_list']

                    loss_ = loss_ * torch.tensor(labels_is_valid[i][agent_idx], device=device, dtype=torch.float).view(self.future_frame_num, 1)
                    if labels_is_valid[i][agent_idx].sum() > utils.eps and should_train:
                        loss[i] += loss_.sum() / labels_is_valid[i][agent_idx].sum()
                        loss[i] += F.nll_loss(pred_probs[i, agent_idx].unsqueeze(0), torch.tensor([argmin], device=device))
                        valid_agent_num += 1

                # print('valid_agent_num', valid_agent_num)

                if valid_agent_num > 0:
                    loss[i] = loss[i] / valid_agent_num

                if self.rebalance_prob:
                    print(self.rebalance_prob_values)

        results = dict(
            pred_outputs=utils.to_numpy(outputs),
            pred_probs=utils.to_numpy(pred_probs),
        )
        return loss.mean(), results, None

    def forward(self,
                mapping: List[Dict],
                batch_size,
                lane_states_batch: List[Tensor],
                inputs: Tensor,
                inputs_lengths: List[int],
                hidden_states: Tensor,
                device,
                labels,
                labels_is_valid,
                agents_indices=None,
                agents=None,
                normalizers=None,
                **kwargs,
                ):
        """
        :param lane_states_batch: each value in list is hidden states of lanes (value shape ['lane num', hidden_size])
        :param inputs: hidden states of all elements before encoding by global graph (shape [batch_size, 'element num', hidden_size])
        :param inputs_lengths: valid element number of each example
        :param hidden_states: hidden states of all elements after encoding by global graph (shape [batch_size, 'element num', hidden_size])
        """
        loss = torch.zeros(batch_size, device=device)
        DE = np.zeros([batch_size, self.future_frame_num])

        if self.variety_loss:
            if self.rebalance_prob:
                with torch.no_grad():
                    return self.forward_variety_loss(mapping, hidden_states, batch_size, inputs, inputs_lengths, labels_is_valid, loss, DE, device, labels,
                                                     agents_indices=agents_indices, agents=agents)
            else:
                return self.forward_variety_loss(mapping, hidden_states, batch_size, inputs, inputs_lengths, labels_is_valid, loss, DE, device, labels,
                                                 agents_indices=agents_indices, agents=agents)
        elif self.dense_decoding:
            return self.forward_dense_decoding(mapping, hidden_states, batch_size, inputs, inputs_lengths, labels_is_valid, loss, DE, device, labels,
                                               agents_indices=agents_indices, agents=agents)
        else:
            assert False
