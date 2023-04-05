import argparse
import inspect
import json
import math
import multiprocessing
import os
import pickle
import random
import subprocess
import sys
import time
import pdb
from collections import defaultdict
from multiprocessing import Process
from random import randint
from typing import Dict, List, Tuple, NamedTuple, Any, Union, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.path import Path
from matplotlib.pyplot import MultipleLocator
from torch import Tensor
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

eps = 1e-5


def get_dis(points: np.ndarray, point_label):
    return np.sqrt(np.square((points[:, 0] - point_label[0])) + np.square((points[:, 1] - point_label[1])))


def get_dis_point2point(point, point_=(0.0, 0.0)):
    return np.sqrt(np.square((point[0] - point_[0])) + np.square((point[1] - point_[1])))


def get_angle(x, y):
    return math.atan2(y, x)


def rotate(x, y, angle):
    res_x = x * math.cos(angle) - y * math.sin(angle)
    res_y = x * math.sin(angle) + y * math.cos(angle)
    return res_x, res_y


def rotate_(x, y, cos, sin):
    res_x = x * cos - y * sin
    res_y = x * sin + y * cos
    return res_x, res_y


def larger(a, b):
    return a > b + eps


def equal(a, b):
    return True if abs(a - b) < eps else False


def load_model(model, state_dict, prefix=''):
    if len(prefix) > 0 and prefix[-1] != '.':
        prefix = f'{prefix}.'

    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix)

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, json.dumps(missing_keys, indent=4)))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, json.dumps(unexpected_keys, indent=4)))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


def merge_tensors(tensors: List[torch.Tensor], device, hidden_size=128) -> Tuple[Tensor, List[int]]:
    """
    merge a list of tensors into a tensor
    """
    lengths = []
    for tensor in tensors:
        lengths.append(tensor.shape[0] if tensor is not None else 0)
    res = torch.zeros([len(tensors), max(lengths), hidden_size], device=device)
    for i, tensor in enumerate(tensors):
        if tensor is not None:
            res[i][:tensor.shape[0]] = tensor
    return res, lengths


def de_merge_tensors(tensor: Tensor, lengths):
    return [tensor[i, :lengths[i]] for i in range(len(lengths))]


def get_one_subdivide_polygon(polygon):
    new_polygon = []
    for i, point in enumerate(polygon):
        if i > 0:
            new_polygon.append((polygon[i - 1] + polygon[i]) / 2)
        new_polygon.append(point)
    return new_polygon


def get_subdivide_polygons(polygon, threshold=2.0):
    if len(polygon) == 1:
        polygon = [polygon[0], polygon[0]]
    elif len(polygon) % 2 == 1:
        polygon = list(polygon)
        polygon = polygon[:len(polygon) // 2] + polygon[-(len(polygon) // 2):]
    assert_(len(polygon) >= 2)

    def get_dis(point_a, point_b):
        return np.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)

    def get_average_dis(polygon):
        average_dis = 0
        for i, point in enumerate(polygon):
            if i > 0:
                average_dis += get_dis(point, point_pre)
            point_pre = point
        average_dis /= len(polygon) - 1
        return average_dis

    average_dis = get_average_dis(polygon)

    if average_dis > threshold:
        length = len(polygon)
        point_a = polygon[length // 2 - 1]
        point_b = polygon[length // 2]
        point_mid = (point_a + point_b) / 2
        polygon_a = polygon[:length // 2]
        polygon_a = get_one_subdivide_polygon(polygon_a)
        polygon_a = polygon_a + [point_mid]
        polygon_b = polygon[length // 2:]
        polygon_b = get_one_subdivide_polygon(polygon_b)
        polygon_b = [point_mid] + polygon_b
        assert_(len(polygon) == len(polygon_a))
        # print('polygon', np.array(polygon), 'polygon_a',np.array(polygon_a), average_dis, get_average_dis(polygon_a))
        return get_subdivide_polygons(polygon_a) + get_subdivide_polygons(polygon_b)
    else:
        return [polygon]


def assert_(satisfied, info=None):
    if not satisfied:
        if info is not None:
            print(info)
        print(sys._getframe().f_code.co_filename, sys._getframe().f_back.f_lineno)
    assert satisfied


def get_color_text(text, color='red'):
    if color == 'red':
        return "\033[31m" + text + "\033[0m"
    else:
        assert False


def get_dis_point_2_points(point, points):
    assert points.ndim == 2
    return np.sqrt(np.square(points[:, 0] - point[0]) + np.square(points[:, 1] - point[1]))


class Normalizer:
    def __init__(self, x, y, yaw):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.origin = rotate(0.0 - x, 0.0 - y, yaw)

    def __call__(self, points, reverse=False):
        points = np.array(points)
        origin_shape = points.shape
        if points.shape == (2,):
            points.shape = (1, 2)
        assert len(points.shape) <= 3
        if len(points.shape) == 3:
            for each in points:
                each[:] = self.__call__(each, reverse)
        else:
            assert len(points.shape) == 2
            for point in points:
                if reverse:
                    point[0], point[1] = rotate(point[0] - self.origin[0],
                                                point[1] - self.origin[1], -self.yaw)
                else:
                    point[0], point[1] = rotate(point[0] - self.x,
                                                point[1] - self.y, self.yaw)

        if origin_shape == (2,):
            points.shape = (2,)
        return points


def to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor.copy()
    return tensor.detach().cpu().numpy()


def past_to_agent_matrix(past_boxes,
                         past_trajs_is_valid,
                         categories,
                         scale=0.03):
    agent_num = len(past_boxes)
    agent_matrix = []
    slices = []
    past_frame_num = past_boxes.shape[1]
    for i in range(agent_num):
        start = len(agent_matrix)
        for j in range(past_frame_num):
            if past_trajs_is_valid[i, j]:
                vector = np.zeros(128)

                cur = 0
                assert past_frame_num <= 10
                vector[cur + j] = 1

                cur = 10
                type_int = categories[i, j]
                if type_int >= 0:
                    vector[cur + type_int] = 1

                cur = 20
                k = 7
                vector[cur:cur + k] = past_boxes[i, j, :k]
                vector[cur:cur + 3] *= scale

                if np.fabs(past_boxes[i, j, :k]).max() > 200:
                    print('too large', past_boxes[i, j, :k])

                agent_matrix.append(vector)

        if len(agent_matrix) > start:
            slices.append(slice(start, len(agent_matrix)))

    return np.array(agent_matrix, dtype=np.float32), slices


def get_transform_and_rotate(point, translation, rotation, reverse=False):
    dim = 3
    if point.shape == (2,):
        dim = 2
        point = np.array(point.tolist() + [0.0])

    from pyquaternion import Quaternion
    if reverse:
        quaternion = Quaternion(rotation).inverse
        point = point - translation
        point = np.dot(quaternion.rotation_matrix, point)
    else:
        quaternion = Quaternion(rotation)
        point = np.dot(quaternion.rotation_matrix, point)
        point = point + translation

    if dim == 2:
        point = point[:2].copy()

    return point


def get_box_from_array(array):
    # array is SECOND format
    yaw = -(array[6] + np.pi / 2)
    # cheatbook: yaw to Quaternion
    rotation = Quaternion(axis=[0, 0, 1], radians=yaw)
    return Box(array[:3], array[3:6], rotation)


def get_array_from_box(box):
    assert isinstance(box, Box)
    # array is SECOND format
    array = box.center.tolist() + box.wlh.tolist() + [-box.orientation.yaw_pitch_roll[0] - np.pi / 2]
    return np.array(array)


def get_SECOND_array_from_box(box):
    assert isinstance(box, Box)
    # array is SECOND format
    array = box.center.tolist() + box.wlh.tolist() + [-box.orientation.yaw_pitch_roll[0] - np.pi / 2]
    return np.array(array)


def get_transform_and_rotate_box(box, translation, rotation, reverse=False):
    assert isinstance(box, Box)
    box = box.copy()

    if isinstance(rotation, tuple):
        print('warning in get_transform_and_rotate_box', rotation, translation)
        rotation = rotation[0]
        translation = translation[0]

    if reverse:
        box.translate(-np.array(translation))
        box.rotate(Quaternion(rotation).inverse)
    else:
        box.rotate(Quaternion(rotation))
        box.translate(np.array(translation))
    return box


def get_gt_past_future_trajs(instance_idx_2_labels):
    gt_past_trajs = []
    gt_past_trajs_is_valid = []
    gt_future_trajs = []
    gt_future_trajs_is_valid = []
    gt_categories = []
    for instance_idx in instance_idx_2_labels:
        def run(future_traj=None,
                future_traj_is_valid=None,
                past_traj=None,
                past_traj_is_valid=None,
                category=None,
                past_boxes=None,
                **kwargs):
            gt_past_trajs.append(past_traj)
            gt_past_trajs_is_valid.append(past_traj_is_valid)
            gt_future_trajs.append(future_traj)
            gt_future_trajs_is_valid.append(future_traj_is_valid)
            gt_categories.append(category)

        run(**instance_idx_2_labels[instance_idx])

    return np.array(gt_past_trajs), np.array(gt_past_trajs_is_valid), np.array(gt_future_trajs), np.array(
        gt_future_trajs_is_valid), np.array(gt_categories)


def get_argmin_traj(gt_trajs, gt_trajs_is_valid, pred_traj, pred_traj_is_valid, FDE=False):
    n = len(gt_trajs)
    if n == 0:
        return None, None

    delta: np.ndarray = gt_trajs - pred_traj[np.newaxis, :]  # [n, len, 2]

    delta = np.sqrt((delta * delta).sum(-1))  # [n, len]

    gt_trajs_is_valid = gt_trajs_is_valid * pred_traj_is_valid[np.newaxis, :]
    delta = delta * gt_trajs_is_valid

    if FDE:
        valid = gt_trajs_is_valid[:, -1] > eps  # [n]
    else:
        valid = np.sum(gt_trajs_is_valid, axis=-1) > eps  # [n]

    nonzero = np.nonzero(valid)  # [m]
    assert isinstance(nonzero, tuple)

    delta = delta[nonzero]  # [m, len]
    indices = np.arange(n)[nonzero]

    if len(delta) == 0:
        return None, None

    if FDE:
        delta = delta[:, -1]  # [m]
    else:
        delta = np.sum(delta, axis=-1) / gt_trajs_is_valid.sum(-1)[nonzero]  # [m]

    argmin = np.argmin(delta)

    # if delta[argmin] > 50:
    #     import pdb;pdb.set_trace()

    return delta[argmin], indices[argmin]


def extract_from_instance_idx_2_labels(instance_idx_2_labels, pc_range=None, relative_pred=False,
                                       pred_challenge_instance_ids=None):
    # bev_range = pc_range[[0, 1, 3, 4]]

    labels_list = []
    labels_is_valid_list = []
    past_trajs = []
    past_trajs_is_valid = []
    categories = []
    past_boxes_list = []
    assert len(instance_idx_2_labels) > 0
    for instance_idx in instance_idx_2_labels:
        def run(future_traj=None,
                future_traj_relative=None,
                future_traj_is_valid=None,
                past_traj=None,
                past_traj_is_valid=None,
                category=None,
                past_boxes=None,
                **kwargs):
            if pred_challenge_instance_ids is not None:
                if instance_idx not in pred_challenge_instance_ids:
                    future_traj_is_valid[:] = 0

            if relative_pred:
                labels_list.append(future_traj_relative)
            else:
                labels_list.append(future_traj)
            labels_is_valid_list.append(future_traj_is_valid)
            past_trajs.append(past_traj)
            past_trajs_is_valid.append(past_traj_is_valid)
            categories.append(category)
            past_boxes_list.append(past_boxes)
            # print('past_traj', past_traj)
            # print('past_traj_is_valid', past_traj_is_valid)
            # print('future_traj', future_traj)
            # print('future_traj_relative', future_traj_relative)
            # print('future_traj_is_valid', future_traj_is_valid)
            # print('past_boxes', past_boxes)

        run(**instance_idx_2_labels[instance_idx])

    return dict(
        labels_list=labels_list,
        labels_is_valid_list=labels_is_valid_list,
        past_trajs=past_trajs,
        past_trajs_is_valid=past_trajs_is_valid,
        categories=categories,
        past_boxes_list=past_boxes_list,
    )


def get_format_data_to_save(data_to_save):
    new_data_to_save = defaultdict(dict)

    for scene_id in data_to_save:
        for key, value in data_to_save[scene_id].items():
            if 'agent' in key:
                length = data_to_save[scene_id]['end_index'] - data_to_save[scene_id]['start_index']
                pose = np.zeros((length, 4))
                size = np.zeros((length, 3))
                category = np.zeros(length, dtype=int)
                pose_is_valid = np.zeros(length)
                future_traj = [None for _ in range(length)]
                predicted_future_traj = [None for _ in range(length)]
                past_traj = [None for _ in range(length)]
                tracked_past_traj = [None for _ in range(length)]

                for each in value:
                    def run(a_pose=None,
                            a_size=None,
                            a_index=None,
                            a_future_traj=None,
                            a_predicted_future_traj=None,
                            a_past_traj=None,
                            a_tracked_past_traj=None,
                            a_category=None):
                        pose[a_index] = a_pose
                        size[a_index] = a_size
                        pose_is_valid[a_index] = 1
                        future_traj[a_index] = a_future_traj
                        predicted_future_traj[a_index] = a_predicted_future_traj
                        past_traj[a_index] = a_past_traj
                        tracked_past_traj[a_index] = a_tracked_past_traj
                        try:
                            category[a_index] = a_category
                        except Exception:
                            assert False

                    run(**each)

                category_cnt = np.zeros(7, dtype=int)
                for i, each in enumerate(category):
                    if pose_is_valid[i]:
                        category_cnt[each] += 1

                new_data_to_save[scene_id][key] = dict(
                    pose=pose.tolist(),
                    size=size.tolist(),
                    category=int(np.argmax(category_cnt)),
                    pose_is_valid=pose_is_valid.tolist(),
                    future_traj=future_traj,
                    predicted_future_traj=predicted_future_traj,
                    past_traj=past_traj,
                    tracked_past_traj=tracked_past_traj,
                )
            else:
                new_data_to_save[scene_id][key] = value

    return new_data_to_save


def get_valid_traj(traj, traj_is_valid):
    valid_traj = []
    for i in range(len(traj)):
        if traj_is_valid[i]:
            valid_traj.append(traj[i])
    return np.array(valid_traj)


def lidar_to_global(array, cur_l2e_t=None, cur_l2e_r=None, cur_e2g_t=None, cur_e2g_r=None, **kwargs):
    box = get_box_from_array(array)
    box = get_transform_and_rotate_box(box, cur_l2e_t, cur_l2e_r)
    box = get_transform_and_rotate_box(box, cur_e2g_t, cur_e2g_r)
    return box


def extract_from_track_idx_2_boxes(track_idx_2_boxes, track_scores, track_ids, track_labels, mapping, index):
    tracked_scores = []
    tracked_trajs = []
    past_boxes_list = []
    past_trajs_is_valid = []
    categories = []

    cur_e2g_t = mapping['cur_e2g_t']
    cur_e2g_r = mapping['cur_e2g_r']

    for j, track_idx in enumerate(track_ids):
        traj = []
        past_boxes = []
        past_traj_is_valid = []
        category = []
        score = []

        for i_index in range(index - 2, index + 1):
            if i_index in track_idx_2_boxes[track_idx]:
                # TODO
                # point[2] += box[5] * 0.5

                # l2e_t = frame_index_2_mapping[i_index]['cur_l2e_t']
                # l2e_r = frame_index_2_mapping[i_index]['cur_l2e_r']
                # e2g_t = frame_index_2_mapping[i_index]['cur_e2g_t']
                # e2g_r = frame_index_2_mapping[i_index]['cur_e2g_r']

                if True:
                    box = track_idx_2_boxes[track_idx][i_index]
                    # box = get_transform_and_rotate_box(box, l2e_t, l2e_r)
                    # box = get_transform_and_rotate_box(box, e2g_t, e2g_r)
                    box = get_transform_and_rotate_box(box, cur_e2g_t, cur_e2g_r, reverse=True)
                    traj.append(box.center[:2].copy())
                    past_boxes.append(get_array_from_box(box).copy())
                else:
                    box = track_idx_2_boxes[track_idx][i_index]

                    point = box[:3].copy()

                    point = get_transform_and_rotate(point, cur_l2e_t, cur_l2e_r)

                    traj.append(point[:2].copy())
                    past_boxes.append(np.concatenate((point, box[3:7])))

                past_traj_is_valid.append(1)
                category.append(track_labels[j])
                score.append(track_scores[j])
            else:
                traj.append(np.zeros(2))
                past_boxes.append(np.zeros(7))
                past_traj_is_valid.append(0)
                category.append(-1)
                score.append(0.)

        tracked_scores.append(np.array(score))
        tracked_trajs.append(np.array(traj))
        past_boxes_list.append(np.array(past_boxes))
        past_trajs_is_valid.append(np.array(past_traj_is_valid))
        categories.append(np.array(category))

    return np.array(tracked_scores), np.array(tracked_trajs), np.array(past_boxes_list), np.array(past_trajs_is_valid), np.array(categories)


def get_labels_for_tracked_trajs(tracked_trajs, past_trajs_is_valid,
                                 gt_past_trajs, gt_past_trajs_is_valid,
                                 gt_future_trajs, gt_future_trajs_is_valid,
                                 future_frame_num):
    labels = []
    labels_is_valid = []
    for j in range(len(tracked_trajs)):
        _, argmin = get_argmin_traj(gt_past_trajs, gt_past_trajs_is_valid, tracked_trajs[j], past_trajs_is_valid[j])
        if argmin is not None:
            labels.append(gt_future_trajs[argmin])
            labels_is_valid.append(gt_future_trajs_is_valid[argmin])
        else:
            labels.append(np.zeros((future_frame_num, 2)))
            labels_is_valid.append(np.zeros((future_frame_num)))
    return np.array(labels), np.array(labels_is_valid)


def get_normalizer(past_traj_is_valid, past_boxes):
    last_valid_box = None
    for j in range(len(past_traj_is_valid))[::-1]:
        if past_traj_is_valid[j]:
            last_valid_box = past_boxes[j]
            break
    x, y, yaw = last_valid_box[0], last_valid_box[1], -last_valid_box[6] - np.pi / 2
    return Normalizer(x, y, -yaw)


def check_code(msg):
    if not hasattr(check_code, 'check'):
        check_code.check = True
        print(get_color_text(msg))
    pass


def update_track_idx_2_boxes(track_idx_2_boxes, track_ids, boxes_3d, mapping, index):
    assert len(track_ids) == len(boxes_3d)
    for j, track_idx in enumerate(track_ids):
        track_idx_2_boxes[track_idx][index] = lidar_to_global(boxes_3d[j], **mapping)


def update_track_idx_2_boxes_in_lidar(track_idx_2_boxes, track_ids, boxes_3d, mapping, index):
    assert len(track_ids) == len(boxes_3d)
    for j, track_idx in enumerate(track_ids):
        track_idx_2_boxes[track_idx][index] = get_box_from_array(boxes_3d[j])


def get_decoded_boxes(pred_boxes, pc_range, img_metas):
    # follow _active_instances2results

    from ..mmdet3d_plugin.core.bbox.util import denormalize_bbox
    bboxes = denormalize_bbox(pred_boxes, pc_range)
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
    bboxes = img_metas[0]['box_type_3d'][0](bboxes, 9)
    return bboxes.to('cpu')


def get_labels_from_pred_logits(pred_logits, num_classes):
    pred_logits = pred_logits.sigmoid()
    _, indexs = pred_logits.max(dim=-1)
    labels = indexs % num_classes
    return to_numpy(labels)


def get_argmax_pred_for_agents(pred_outputs, pred_probs):
    argmax = np.argmax(pred_probs, axis=-1)
    return pred_outputs[np.arange(len(pred_outputs)), argmax]


def img_metas_tracking_to_detection(img_metas, cur_frame, num_frame=None):
    from copy import deepcopy
    img_metas = deepcopy(img_metas)

    def get_single(img_meta):
        img_meta_single = {}
        for key, value in img_meta.items():
            if num_frame is not None:
                assert len(value) == num_frame
            img_meta_single[key] = value[cur_frame]
        return img_meta_single

    img_metas_single = [get_single(each) for each in img_metas]
    return img_metas_single


def tensors_tracking_to_detection(tensors, cur_frame):
    if tensors is None:
        return None
    return [each[cur_frame] for each in tensors]


def get_attention_cfg():
    point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

    return dict(
        type='DetrTransformerDecoderLayer',
        attn_cfgs=[
            dict(
                type='MultiheadAttention',
                embed_dims=256,
                num_heads=8,
                dropout=0.1),
            dict(
                type='Detr3DCrossAtten',
                pc_range=point_cloud_range,
                num_points=1,
                embed_dims=256,
            )
        ],
        feedforward_channels=512,
        ffn_dropout=0.1,
        operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                         'ffn', 'norm'))


def reference_points_relative_to_lidar(reference_points: np.ndarray, pc_range):
    reference_points = reference_points.copy()
    reference_points[..., 0:1] = reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    reference_points[..., 2:3] = reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
    return reference_points


def reference_points_lidar_to_relative(reference_points: np.ndarray, pc_range):
    reference_points = reference_points.copy()
    reference_points[..., 0:1] = (reference_points[..., 0:1] - pc_range[0]) / (pc_range[3] - pc_range[0])
    reference_points[..., 1:2] = (reference_points[..., 1:2] - pc_range[1]) / (pc_range[4] - pc_range[1])
    reference_points[..., 2:3] = (reference_points[..., 2:3] - pc_range[2]) / (pc_range[5] - pc_range[2])
    return reference_points


def merge_images(image_paths: dict, output_dir, name):
    import mmcv
    paths = [os.path.join('data/nuscenes', path) for (camera, path) in image_paths.items()]
    images = []
    for path in paths:
        images.append(mmcv.imread(path))

    camera_types = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_FRONT_LEFT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT',
    ]
    assert len(images) == 6

    height = images[0].shape[0]
    width = images[0].shape[1]
    interval = 30
    # dtype is uint8
    merged = np.ones((height * 2 + interval, width * 3 + 2 * interval, 3), dtype=images[0].dtype) * 255

    def set_image(merged, image, w, h, height, width):
        assert image.shape == (height, width, 3)
        merged[h:h + height, w: w + width] = image

    def set_row_of_images(merged, images, offset, height, width, interval):
        for i in range(3):
            w = i * (width + interval)
            set_image(merged, images[i], w, offset, height, width)

    set_row_of_images(merged, [images[2], images[0], images[1]], 0, height, width, interval)
    set_row_of_images(merged, [images[5], images[3], images[4]], height + interval, height, width, interval)
    # set_row_of_images(merged, [images[4][:, ::-1, :], images[3][:, ::-1, :], images[5][:, ::-1, :]], height + interval, height, width, interval)

    merged = mmcv.imresize(merged, (merged.shape[1] // 4, merged.shape[0] // 4))
    # mmcv.imwrite(merged, os.path.join(output_dir, f'{name}.png'))


tracking_class_names = [
    'car', 'truck', 'bus', 'trailer',
    'motorcycle', 'bicycle', 'pedestrian',
]

class_name_to_index = {}
for index, each in enumerate(tracking_class_names):
    class_name_to_index[each] = index


def get_array_from_dict_while_from_global_to_lidar(box_dict, mapping):
    from nuscenes.utils.data_classes import Box
    # see site-packages/nuscenes/nuscenes.py:368
    box = Box(box_dict['translation'], box_dict['size'], Quaternion(box_dict['rotation']))
    box = get_transform_and_rotate_box(box, mapping['cur_e2g_t'], mapping['cur_e2g_r'], reverse=True)
    box = get_transform_and_rotate_box(box, mapping['cur_l2e_t'], mapping['cur_l2e_r'], reverse=True)
    array = get_SECOND_array_from_box(box)
    return array


def pos2posemb(pos, num_pos_feats=64, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    posemb = pos[..., None] / dim_t
    posemb = torch.stack((posemb[..., 0::2].sin(), posemb[..., 1::2].cos()), dim=-1).flatten(-3)
    return posemb


def load_proposals(timestamp, proposal_dir, is_train):
    import pickle
    s = 'train' if is_train else 'val'
    with open(os.path.join(proposal_dir, f'{s}.{timestamp}.pkl'), 'rb') as f:
        proposals = pickle.load(f)
    return proposals


def set_pc_range_to_unit_range(points, pc_range, x=0, y=1, z=2, reverse=False):
    if reverse:
        points[..., x:x + 1] = points[..., x:x + 1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        points[..., y:y + 1] = points[..., y:y + 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
        points[..., z:z + 1] = points[..., z:z + 1] * (pc_range[5] - pc_range[2]) + pc_range[2]
    else:
        points[..., x:x + 1] = (points[..., x:x + 1] - pc_range[0]) / (pc_range[3] - pc_range[0])
        points[..., y:y + 1] = (points[..., y:y + 1] - pc_range[1]) / (pc_range[4] - pc_range[1])
        points[..., z:z + 1] = (points[..., z:z + 1] - pc_range[2]) / (pc_range[5] - pc_range[2])


def detection_scores_to_tracking_scoers(detection_scores):
    if False:
        detection_class_names = [
            'car', 'truck', 'bus', 'trailer',
            'motorcycle', 'bicycle', 'pedestrian',
        ]
        tracking_class_names = [
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
            'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]

    return detection_scores[..., [0, 1, 3, 4, 6, 7, 8]]


def get_coordinates_from_track_instance_boxes(boxes):
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if len(boxes.shape) == 1:
        return np.concatenate([boxes[0:2], boxes[4:5]])
    return np.concatenate([boxes[:, 0:2], boxes[:, 4:5]], axis=-1)
