from typing import Dict, List, Tuple, NamedTuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .predictor_decoder import Decoder, DecoderResCat
from .predictor_lib import MLP, GlobalGraph, LayerNorm, SubGraph, CrossAttention, GlobalGraphRes
from .. import utils as utils


class NewSubGraph(nn.Module):

    def __init__(self, hidden_size, depth=3):
        super(NewSubGraph, self).__init__()
        self.layers = nn.ModuleList([MLP(hidden_size, hidden_size // 2) for _ in range(depth)])
        if True:
            self.layer_0 = MLP(hidden_size)
            self.layers = nn.ModuleList([GlobalGraph(hidden_size, num_attention_heads=2) for _ in range(depth)])
            self.layers_2 = nn.ModuleList([LayerNorm(hidden_size) for _ in range(depth)])
            self.layers_3 = nn.ModuleList([LayerNorm(hidden_size) for _ in range(depth)])
            self.layers_4 = nn.ModuleList([GlobalGraph(hidden_size) for _ in range(depth)])
            if True:
                self.layer_0_again = MLP(hidden_size)

    def forward(self, input_list: list):
        batch_size = len(input_list)
        device = input_list[0].device
        hidden_states, lengths = utils.merge_tensors(input_list, device)
        hidden_size = hidden_states.shape[2]
        max_vector_num = hidden_states.shape[1]

        if True:
            attention_mask = torch.zeros([batch_size, max_vector_num, max_vector_num], device=device)
            hidden_states = self.layer_0(hidden_states)

            if True:
                hidden_states = self.layer_0_again(hidden_states)
            for i in range(batch_size):
                assert lengths[i] > 0
                attention_mask[i, :lengths[i], :lengths[i]].fill_(1)

            for layer_index, layer in enumerate(self.layers):
                temp = hidden_states
                # hidden_states = layer(hidden_states, attention_mask)
                # hidden_states = self.layers_2[layer_index](hidden_states)
                # hidden_states = F.relu(hidden_states) + temp
                hidden_states = layer(hidden_states, attention_mask)
                hidden_states = F.relu(hidden_states)
                hidden_states = hidden_states + temp
                hidden_states = self.layers_2[layer_index](hidden_states)

        return torch.max(hidden_states, dim=1)[0], torch.cat(utils.de_merge_tensors(hidden_states, lengths))


class VectorNet(nn.Module):
    r"""
    VectorNet

    It has two main components, sub graph and global graph.

    Sub graph encodes a polyline as a single vector.
    """

    def __init__(self,
                 hidden_size=128,
                 laneGCN=False,
                 decoder=None,
                 ):
        super(VectorNet, self).__init__()

        self.sub_graph = SubGraph(hidden_size)

        if True:
            self.point_level_sub_graph = NewSubGraph(hidden_size)
            self.point_level_cross_attention = CrossAttention(hidden_size)

        self.global_graph = GlobalGraph(hidden_size)

        self.laneGCN = laneGCN
        if laneGCN:
            self.laneGCN_A2L = CrossAttention(hidden_size)
            self.laneGCN_L2L = GlobalGraphRes(hidden_size)
            self.laneGCN_L2A = CrossAttention(hidden_size)

        self.sub_graph_map = SubGraph(hidden_size)

        self.decoder = Decoder(self, **decoder)

    def forward_encode_sub_graph(self,
                                 mapping: List[Dict],
                                 lane_matrix: List[np.ndarray],
                                 polyline_spans: List[List[slice]],
                                 device,
                                 batch_size,
                                 agents_batch=None,
                                 agent_matrix=None,
                                 agent_matrix_slices=None,
                                 **kwargs) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """
        :param lane_matrix: each value in list is vectors of all element (shape [-1, 128])
        :param polyline_spans: vectors of i_th element is matrix[polyline_spans[i]]
        :return: hidden states of all elements and hidden states of lanes
        """
        assert batch_size == 1, batch_size

        input_list_list = []
        # TODO(cyrushx): This is not used? Is it because input_list_list includes map data as well?
        # Yes, input_list_list includes map data, this will be used in the future release.
        map_input_list_list = []
        lane_states_batch = None
        for i in range(batch_size):
            input_list = []
            map_input_list = []
            # map_start_polyline_idx = mapping[i]['map_start_polyline_idx']
            map_start_polyline_idx = 0
            for j, polyline_span in enumerate(polyline_spans[i]):
                tensor = torch.tensor(lane_matrix[i][polyline_span], device=device)
                input_list.append(tensor)
                if j >= map_start_polyline_idx:
                    map_input_list.append(tensor)

            input_list_list.append(input_list)
            map_input_list_list.append(map_input_list)

        if agent_matrix is not None:
            agent_input_list_list = []
            for i in range(batch_size):
                input_list = []
                for j, each in enumerate(agent_matrix_slices[i]):
                    tensor = torch.tensor(agent_matrix[i][each], device=device, dtype=torch.float)
                    input_list.append(tensor)
                agent_input_list_list.append(input_list)

            if agents_batch is None:
                agents_batch = []
                for i in range(batch_size):
                    assert len(agent_input_list_list[i]) > 0
                    a, _ = self.point_level_sub_graph(agent_input_list_list[i])
                    agents_batch.append(a)
            else:
                for i in range(batch_size):
                    assert len(agent_input_list_list[i]) > 0
                    a, _ = self.point_level_sub_graph(agent_input_list_list[i])
                    assert len(agents_batch[i]) == len(a)
                    agents_batch[i] = agents_batch[i] + a
        else:
            assert agents_batch is not None

        if True:
            lane_states_batch = []
            for i in range(batch_size):
                assert len(map_input_list_list[i]) > 0
                a, _ = self.point_level_sub_graph(map_input_list_list[i])
                lane_states_batch.append(a)

        element_states_batch = []
        if self.laneGCN:
            for i in range(batch_size):
                # map_start_polyline_idx = mapping[i]['map_start_polyline_idx']
                agents = agents_batch[i]
                lanes = lane_states_batch[i]
                assert len(agents) > 0 and len(lanes) > 0

                lanes = lanes + self.laneGCN_A2L(lanes.unsqueeze(0), torch.cat([lanes, agents]).unsqueeze(0)).squeeze(0)
                # else:
                #     lanes = lanes + self.laneGCN_A2L(lanes.unsqueeze(0), agents.unsqueeze(0)).squeeze(0)
                #     lanes = lanes + self.laneGCN_L2L(lanes.unsqueeze(0)).squeeze(0)
                #     agents = agents + self.laneGCN_L2A(agents.unsqueeze(0), lanes.unsqueeze(0)).squeeze(0)
                element_states_batch.append(torch.cat([agents, lanes]))
        else:
            for i in range(batch_size):
                agents = agents_batch[i]
                lanes = lane_states_batch[i]
                element_states_batch.append(torch.cat([agents, lanes]))

        return element_states_batch, lane_states_batch, agents_batch

    # @profile
    def forward(self,
                pred_matrix=None,
                polyline_spans=None,
                mapping=None,
                labels=None,
                labels_is_valid=None,
                agents: List[Tensor] = None,
                agents_indices=None,
                device=None,
                **kwargs,
                ):
        import time
        global starttime
        starttime = time.time()

        lane_matrix = pred_matrix
        polyline_spans = polyline_spans
        mapping = mapping
        if 'work_dir' in mapping and np.random.randint(20) == 0:
            print(f'work_dir {mapping["work_dir"]}')
        if np.random.randint(0, 50) == 0:
            print('index', mapping[0]['index'], device)
        if agents is not None:
            agents = [each[:, :128] for each in agents]
        # TODO(cyrushx): Can you explain the structure of polyline spans?
        # vectors of i_th element is matrix[polyline_spans[i]]

        batch_size = len(lane_matrix)
        # for i in range(batch_size):
        # polyline_spans[i] = [slice(polyline_span[0], polyline_span[1]) for polyline_span in polyline_spans[i]]

        element_states_batch, lane_states_batch, agents = self.forward_encode_sub_graph(mapping, lane_matrix, polyline_spans, device, batch_size,
                                                                                        agents_batch=agents, **kwargs)

        inputs, inputs_lengths = utils.merge_tensors(element_states_batch, device=device)
        max_poly_num = max(inputs_lengths)
        attention_mask = torch.zeros([batch_size, max_poly_num, max_poly_num], device=device)
        for i, length in enumerate(inputs_lengths):
            attention_mask[i][:length][:length].fill_(1)

        hidden_states = self.global_graph(inputs, attention_mask, mapping)

        return self.decoder(mapping, batch_size, lane_states_batch, inputs, inputs_lengths, hidden_states, device,
                            labels, labels_is_valid, agents_indices, agents=agents, **kwargs)
