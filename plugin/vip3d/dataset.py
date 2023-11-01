import copy
import math
import os
import pickle
import tempfile
from os import path as osp
from typing import Dict, List, Tuple

import mmcv
import numpy as np
import pyquaternion
import torch
from mmdet.datasets import DATASETS
from nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap, locations
from nuscenes.prediction import PredictHelper
from nuscenes.utils.data_classes import Box as NuScenesBox
from pyquaternion import Quaternion
from torch.utils.data import Dataset

from mmdet3d.core import show_result
from mmdet3d.core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
from mmdet3d.core.bbox import get_box_type
from mmdet3d.datasets.pipelines import Compose
from . import utils


@DATASETS.register_module()
class NuScenesTrackDatasetRadar(Dataset):
    r"""NuScenes Dataset.

    This class serves as the API for experiments on the NuScenes Dataset.

    Please refer to `NuScenes Dataset <https://www.nuscenes.org/download>`_
    for data downloading.

    Args:
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        data_root (str): Path of dataset root.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        load_interval (int, optional): Interval of loading the dataset. It is
            used to uniformly sample the dataset. Defaults to 1.
        with_velocity (bool, optional): Whether include velocity prediction
            into the experiments. Defaults to True.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes.
            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        eval_version (bool, optional): Configuration version of evaluation.
            Defaults to  'detection_cvpr_2019'.
        use_valid_flag (bool): Whether to use `use_valid_flag` key in the info
            file as mask to filter gt_boxes and gt_names. Defaults to False.
    """
    NameMapping = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'
    }
    DefaultAttribute = {
        'car': 'vehicle.parked',
        'pedestrian': 'pedestrian.moving',
        'trailer': 'vehicle.parked',
        'truck': 'vehicle.parked',
        'bus': 'vehicle.moving',
        'motorcycle': 'cycle.without_rider',
        'construction_vehicle': 'vehicle.parked',
        'bicycle': 'cycle.without_rider',
        'barrier': '',
        'traffic_cone': '',
    }
    AttrMapping = {
        'cycle.with_rider': 0,
        'cycle.without_rider': 1,
        'pedestrian.moving': 2,
        'pedestrian.standing': 3,
        'pedestrian.sitting_lying_down': 4,
        'vehicle.moving': 5,
        'vehicle.parked': 6,
        'vehicle.stopped': 7,
    }
    AttrMapping_rev = [
        'cycle.with_rider',
        'cycle.without_rider',
        'pedestrian.moving',
        'pedestrian.standing',
        'pedestrian.sitting_lying_down',
        'vehicle.moving',
        'vehicle.parked',
        'vehicle.stopped',
    ]
    CLASSES = ['car', 'truck', 'bus', 'trailer',
               'motorcycle', 'bicycle', 'pedestrian']

    def __init__(self,
                 ann_file,
                 pipeline_single=None,
                 pipeline_post=None,
                 data_root=None,
                 classes=None,
                 load_interval=1,
                 with_velocity=True,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 eval_version='detection_cvpr_2019',
                 sample_mode='fixed_interval',
                 sample_interval=1,
                 num_frames_per_sample=3,
                 use_valid_flag=True,
                 do_pred=False,
                 calc_prediction_metric=False,
                 generate_nuscenes_prediction_infos_val=False,
                 **kwargs,
                 ):
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        super().__init__()

        self.data_root = data_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.modality = modality
        self.filter_empty_gt = filter_empty_gt
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)

        self.CLASSES = self.get_classes(classes)
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        self.data_infos = self.load_annotations(self.ann_file)

        self.sample_mode = sample_mode
        self.sample_interval = sample_interval
        self.num_frames_per_sample = num_frames_per_sample
        if not self.test_mode:
            self.training_frame_num = num_frames_per_sample

        # if not self.test_mode:
        #     self.num_frames_per_sample += 1
        self.num_samples = len(self.data_infos) - (self.num_frames_per_sample - 1) * \
                           self.sample_interval

        if pipeline_single is not None:
            self.pipeline_single = Compose(pipeline_single)

        if pipeline_post is not None:
            self.pipeline_post = Compose(pipeline_post)

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

        self.with_velocity = with_velocity
        self.eval_version = eval_version
        from nuscenes.eval.detection.config import config_factory
        self.eval_detection_configs = config_factory(self.eval_version)
        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )
        self.do_pred = do_pred
        self.generate_nuscenes_prediction_infos_val = generate_nuscenes_prediction_infos_val

    def prepare_nuscenes(self):
        self.nuscenes = NuScenes('v1.0-trainval/', dataroot=self.data_root)
        # self.nuscenes = NuScenes('v1.0-mini', dataroot=data_root)
        self.helper = PredictHelper(self.nuscenes)
        self.maps = load_all_maps(self.helper)

    def generate_prediction(self, data_detection, cur_info, index):
        assert self.test_mode
        assert len(self) == len(self.data_infos)

        self.add_labels_for_test_info(data_detection, index)

        instance_inds = data_detection['instance_inds']
        gt_bboxes_3d = data_detection['gt_bboxes_3d'].tensor.numpy()
        gt_labels_3d = data_detection['gt_labels_3d']
        sample_token = cur_info['token']

        l2e_r = cur_info['lidar2ego_rotation']
        l2e_t = cur_info['lidar2ego_translation']
        e2g_r = cur_info['ego2global_rotation']
        e2g_t = cur_info['ego2global_translation']

        new_gt_bboxes_3d = []
        for box_idx in range(len(instance_inds)):
            box = utils.get_box_from_array(gt_bboxes_3d[box_idx])
            box = utils.get_transform_and_rotate_box(box, l2e_t, l2e_r)
            box = utils.get_transform_and_rotate_box(box, e2g_t, e2g_r)
            utils.fix_box_from_array = True
            array = utils.get_array_from_box(box)
            new_gt_bboxes_3d.append(array)
        gt_bboxes_3d = np.array(new_gt_bboxes_3d)

        if not hasattr(self, 'nuscenes_prediction_infos_val'):
            from collections import OrderedDict
            self.nuscenes_prediction_infos_val = OrderedDict()

        self.nuscenes_prediction_infos_val[sample_token] = dict(
            sample_token=sample_token,
            timestamp=cur_info['timestamp'],
            instance_inds=instance_inds.tolist(),
            gt_bboxes_3d=gt_bboxes_3d.tolist(),
            gt_labels_3d=gt_labels_3d.tolist()
        )

        print('length of nuscenes_prediction_infos_val', len(self.nuscenes_prediction_infos_val))

        if index == len(self.data_infos) - 1:
            with open(os.path.join(self.data_root, 'nuscenes_prediction_infos_val.json'), 'w') as f:
                import json
                json.dump(self.nuscenes_prediction_infos_val, f, indent=4)

    def add_labels_for_test_info(self, data_i, index):
        from mmcv.utils import build_from_cfg
        from mmdet.datasets.builder import PIPELINES

        point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        range_filter = build_from_cfg(dict(type='InstanceRangeFilter', point_cloud_range=point_cloud_range), PIPELINES)
        data_i['ann_info'] = self.get_ann_info(index)
        data_i['gt_bboxes_3d'] = data_i['ann_info']['gt_bboxes_3d']
        data_i['gt_labels_3d'] = data_i['ann_info']['gt_labels_3d']
        range_filter(data_i)
        data_i['instance_inds'] = data_i['ann_info']['instance_inds']

    def get_transform_and_rotate(self, point, translation, rotation, reverse=False):
        if reverse:
            quaternion = Quaternion(rotation).inverse
            point = point - translation
            point = np.dot(quaternion.rotation_matrix, point)
        else:
            quaternion = Quaternion(rotation)
            point = np.dot(quaternion.rotation_matrix, point)
            point = point + translation
        return point

    def prepare_data_history(self, start, end, interval):
        ret = None
        for i in range(start, end, interval):
            if not (0 <= i < len(self.data_infos)):
                return None

            if self.test_mode:
                from mmcv.utils import build_from_cfg
                from mmdet.datasets.builder import PIPELINES
                point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
                range_filter = build_from_cfg(dict(type='InstanceRangeFilter', point_cloud_range=point_cloud_range), PIPELINES)
                data_i = self.prepare_test_data_single(i)
                data_i['ann_info'] = self.get_ann_info(i)
                data_i['gt_bboxes_3d'] = data_i['ann_info']['gt_bboxes_3d']
                data_i['gt_labels_3d'] = data_i['ann_info']['gt_labels_3d']
                range_filter(data_i)
                data_i['instance_inds'] = data_i['ann_info']['instance_inds']
            else:
                data_i = self.prepare_train_data_single(i)

            if data_i is None:
                return None

            if ret is None:
                ret = {key: [] for key in data_i.keys()}

            for key, value in data_i.items():
                ret[key].append(value)
        return ret

    def get_pred_agents(self, results, start, end, interval, mapping, data_history=None):
        future_frame_num = 12
        past_frame_num = end - start
        instance_idx_2_labels = {}
        r_index_2_rotation_and_transform = {}

        cur_info = self.data_infos[end - 1]
        cur_l2e_r = cur_info['lidar2ego_rotation']
        cur_l2e_t = cur_info['lidar2ego_translation']
        cur_e2g_r = cur_info['ego2global_rotation']
        cur_e2g_t = cur_info['ego2global_translation']

        same_scene = self.is_the_same_scene(start, end, future_frame_num)

        data_history = self.prepare_data_history(start, end + future_frame_num, interval)

        if same_scene and data_history is not None:

            for i in range(start, end + future_frame_num):
                assert 0 <= i < len(self.data_infos)
                # i = max(i, 0)
                # i = min(i, len(self.data_infos) - 1)

                info = self.data_infos[i]
                # filter out bbox containing no points
                # if self.use_valid_flag:
                #     mask = info['valid_flag']
                # else:
                #     mask = info['num_lidar_pts'] > 0
                l2e_r = info['lidar2ego_rotation']
                l2e_t = info['lidar2ego_translation']
                e2g_r = info['ego2global_rotation']
                e2g_t = info['ego2global_translation']

                if i < end:
                    r_index_2_rotation_and_transform[i - start] = dict(
                        cur_l2e_r=l2e_r,
                        cur_l2e_t=l2e_t,
                        cur_e2g_r=e2g_r,
                        cur_e2g_t=e2g_t,
                    )

                assert end + future_frame_num - start == len(data_history['instance_inds'])

                # instance_inds = data_history['ann_info'][i - start]['instance_inds']
                instance_inds = data_history['instance_inds'][i - start]

                gt_bboxes_3d = data_history['gt_bboxes_3d'][i - start].tensor.numpy()
                gt_labels_3d = data_history['gt_labels_3d'][i - start]
                assert len(instance_inds) == len(gt_bboxes_3d) == len(gt_labels_3d)
                if not self.test_mode:
                    assert len(instance_inds) > 0

                # gt_bboxes_3d = info['gt_boxes'][mask]
                # instance_inds = np.array(info['instance_inds'], dtype=np.int)[mask]

                # def get_gt_labels_3d():
                #     gt_names_3d = info['gt_names'][mask]
                #     gt_labels_3d = []
                #     for cat in gt_names_3d:
                #         if cat in self.CLASSES:
                #             gt_labels_3d.append(self.CLASSES.index(cat))
                #         else:
                #             gt_labels_3d.append(-1)
                #     gt_labels_3d = np.array(gt_labels_3d)
                #     return gt_labels_3d
                #
                # gt_labels_3d = get_gt_labels_3d()

                for box_idx, instance_idx in enumerate(instance_inds):
                    assert instance_idx != -1
                    if instance_idx not in instance_idx_2_labels:
                        if i >= end:
                            continue

                        instance_idx_2_labels[instance_idx] = dict(
                            future_traj=np.zeros((future_frame_num, 2), dtype=np.float32),
                            future_traj_relative=np.zeros((future_frame_num, 2), dtype=np.float32),
                            future_traj_is_valid=np.zeros(future_frame_num, dtype=np.int32),
                            past_traj=np.zeros((past_frame_num, 2), dtype=np.float32),
                            past_traj_is_valid=np.zeros(past_frame_num, dtype=np.int32),
                            category=np.zeros(past_frame_num, dtype=np.int32),
                            past_boxes=np.zeros((past_frame_num, 7), dtype=np.float32),
                        )

                    def run(future_traj=None,
                            future_traj_relative=None,
                            future_traj_is_valid=None,
                            past_traj=None,
                            past_traj_is_valid=None,
                            category=None,
                            past_boxes=None,
                            **kwargs):
                        box = utils.get_box_from_array(gt_bboxes_3d[box_idx])
                        box = utils.get_transform_and_rotate_box(box, l2e_t, l2e_r)
                        box = utils.get_transform_and_rotate_box(box, e2g_t, e2g_r)
                        box = utils.get_transform_and_rotate_box(box, cur_e2g_t, cur_e2g_r, reverse=True)
                        point = box.center

                        if i < end:
                            past_traj[i - start] = point[0], point[1]
                            past_traj_is_valid[i - start] = 1
                            category[i - start] = gt_labels_3d[box_idx]

                            past_boxes[i - start, :7] = utils.get_array_from_box(box)
                        else:
                            future_traj[i - end] = point[0], point[1]

                            normalizer = utils.get_normalizer(past_traj_is_valid, past_boxes)

                            future_traj_relative[i - end] = normalizer(point[:2])
                            future_traj_is_valid[i - end] = 1
                            if not same_scene:
                                future_traj_is_valid[i - end] = 0

                            # if not same_scene and i > self.last_index:
                            #     future_traj_is_valid[i - end] = 0
                        pass

                    run(**instance_idx_2_labels[instance_idx])

        instance_inds = None
        if same_scene and data_history is not None:
            instance_inds = data_history['instance_inds'][end - 1 - start]

        mapping.update(dict(
            cur_l2e_r=cur_l2e_r,
            cur_l2e_t=cur_l2e_t,
            cur_e2g_r=cur_e2g_r,
            cur_e2g_t=cur_e2g_t,
            r_index_2_rotation_and_transform=r_index_2_rotation_and_transform,
            valid_pred=same_scene and data_history is not None,
            instance_inds=instance_inds,
        ))

        results['instance_idx_2_labels'] = instance_idx_2_labels

    def get_pred_lanes(self, results, start, end, interval, mapping):
        cur_info = self.data_infos[end - 1]
        cur_l2e_r = cur_info['lidar2ego_rotation']
        cur_l2e_t = cur_info['lidar2ego_translation']
        cur_e2g_r = cur_info['ego2global_rotation']
        cur_e2g_t = cur_info['ego2global_translation']

        point = np.array([0.0, 0.0, 0.0])
        point = self.get_transform_and_rotate(point, cur_e2g_t, cur_e2g_r)
        agent_x, agent_y = point[0], point[1]

        helper = self.helper

        # instance_token, sample_token = file_name.split("_")
        sample_token = cur_info['token']

        map_name = helper.get_map_name_from_sample_token(sample_token)

        # sample_annotation = helper.get_sample_annotation(instance_token, sample_token)
        # agent_x, agent_y = sample_annotation['translation'][:2]

        # yaw = quaternion_yaw(Quaternion(sample_annotation['rotation']))
        yaw = 0.0
        angle = -yaw + math.radians(90)

        normalizer = utils.Normalizer(agent_x, agent_y, angle)

        max_dis = 70.0
        visible_y = 30.0
        discretization_resolution_meters = 1

        nuscene_lanes = get_lanes_in_radius(agent_x, agent_y, max_dis, discretization_resolution_meters, self.maps[map_name])

        vectors = []
        polyline_spans = []

        polygons = []

        def get_dis_point2point(point, point_=(0.0, 0.0)):
            return np.sqrt(np.square((point[0] - point_[0])) + np.square((point[1] - point_[1])))

        if True:
            for poses_along_lane in nuscene_lanes.values():
                lane = [pose[:2] for pose in poses_along_lane]
                lane = np.array(lane)
                lane = normalizer(lane)

                lane = np.array([point for point in lane if get_dis_point2point(point, (0.0, visible_y)) < max_dis])
                if len(lane) < 1:
                    continue

                polygons.append(lane)

                start = len(vectors)
                stride = 5
                scale = 0.05
                vector = np.zeros(128)
                for j in range(0, len(lane), stride):

                    cur = 0
                    for k in range(stride + 2):
                        t = min(j + k, len(lane) - 1)
                        vector[cur + 2 * k + 0] = lane[t, 0] * scale
                        vector[cur + 2 * k + 1] = lane[t, 1] * scale

                    # cur = 30
                    # if type[now] != -1:
                    #     assert type[now] < 20
                    #     vector[cur + type[now]] = 1.0

                    cur = 40
                    vector[cur + 0] = j
                    t_float = j
                    vector[cur + 1] = t_float / len(lane)

                    vectors.append(vector)

                if len(vectors) > start:
                    polyline_spans.append(slice(start, len(vectors)))

        if len(polyline_spans) == 0:
            start = len(vectors)
            assert start == 0

            vectors.append(np.zeros(128))
            polyline_spans.append(slice(start, len(vectors)))

            lane = np.zeros([1, 2], dtype=np.float32)
            polygons.append(lane)

        if True:
            info = self.data_infos[end - 1]
            # filter out bbox containing no points
            if self.use_valid_flag:
                mask = info['valid_flag']
            else:
                mask = info['num_lidar_pts'] > 0

        mapping.update(dict(
            lanes=polygons,
            map_name=map_name,
        ))

        results.update(dict(
            pred_matrix=np.array(vectors, dtype=np.float32),
            polyline_spans=polyline_spans,
        ))

    def __len__(self):
        """Return the length of data infos.

        Returns:
            int: Length of data infos.
        """
        return self.num_samples

    @classmethod
    def get_classes(cls, classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        Return:
            list[str]: A list of class names.
        """
        if classes is None:
            return cls.CLASSES

        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        return class_names

    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.data_infos[idx]
        if self.use_valid_flag:
            mask = info['valid_flag']
            gt_names = set(info['gt_names'][mask])
        else:
            gt_names = set(info['gt_names'])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file)
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        if not self.test_mode:
            return data_infos
        return data_infos

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        # index = 5
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
            radar=info['radars'],
        )

        # print('index', index, info['timestamp'] / 1e6)

        if self.data_root not in input_dict['pts_filename']:
            input_dict['pts_filename'] = input_dict['pts_filename'].replace('data/nuscenes/', self.data_root)

        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        l2g_r_mat = (l2e_r_mat.T @ e2g_r_mat.T).T  # [3, 3]
        l2g_t = l2e_t @ e2g_r_mat.T + e2g_t  # [1, 3]

        input_dict.update(
            dict(
                l2g_r_mat=l2g_r_mat.astype(np.float32),
                l2g_t=l2g_t.astype(np.float32)))

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            intrinsics = []
            extrinsics = []
            for cam_type, cam_info in info['cams'].items():
                if self.data_root not in cam_info['data_path']:
                    image_paths.append(cam_info['data_path'].replace('data/nuscenes/', self.data_root))
                else:
                    image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                                  'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)
                intrinsics.append(viewpad)
                extrinsics.append(lidar2cam_rt.T)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    intrinsic=intrinsics,
                    extrinsic=extrinsics,
                ))

        if True:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        instance_inds = np.array(info['instance_inds'], dtype=np.int)[mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            instance_inds=instance_inds)
        return anns_results

    def pre_pipeline(self, results):
        """Initialization before data preparation.

        Args:
            results (dict): Dict before data preprocessing.

                - img_fields (list): Image fields.
                - bbox3d_fields (list): 3D bounding boxes fields.
                - pts_mask_fields (list): Mask fields of points.
                - pts_seg_fields (list): Mask fields of point segments.
                - bbox_fields (list): Fields of bounding boxes.
                - mask_fields (list): Fields of masks.
                - seg_fields (list): Segment fields.
                - box_type_3d (str): 3D box type.
                - box_mode_3d (str): 3D box mode.
        """
        results['img_fields'] = []
        results['bbox3d_fields'] = []
        results['pts_mask_fields'] = []
        results['pts_seg_fields'] = []
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['box_type_3d'] = self.box_type_3d
        results['box_mode_3d'] = self.box_mode_3d

    def _get_sample_range(self, start_idx):

        # take default sampling method for normal dataset.
        assert self.sample_mode in ['fixed_interval', 'random_interval'], 'invalid sample mode: {}'.format(self.sample_mode)
        if self.sample_mode == 'fixed_interval':
            sample_interval = self.sample_interval
        elif self.sample_mode == 'random_interval':
            sample_interval = np.random.randint(1, self.sample_interval + 1)
        default_range = start_idx, start_idx + (self.num_frames_per_sample - 1) * sample_interval + 1, sample_interval
        return default_range

    def pre_continuous_frames(self, start, end, interval=1):
        targets = []
        images = []
        for i in range(start, end, interval):
            img_i, targets_i = self._pre_single_frame(i)
            images.append(img_i)
            targets.append(targets_i)
        return images, targets

    def prepare_train_data_single(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline_single(input_dict)

        example['instance_inds'] = example['ann_info']['instance_inds']
        if self.filter_empty_gt and \
                (example is None or
                 ~(example['gt_labels_3d'] != -1).any()):
            return None
        return example

    def is_the_same_scene(self, start, end, future_frame_num):
        timestamps = []
        for i in range(start, end + future_frame_num):
            if i < 0 or i >= len(self.data_infos):
                return False
            info = self.data_infos[i]
            timestamp = info['timestamp'] / 1e6
            timestamps.append(timestamp)

        for i in range(len(timestamps) - 1):
            if abs(timestamps[i + 1] - timestamps[i]) > 10:
                self.last_index = start + i
                return False

        return True

    def prepare_train_data(self, index):
        start, end, interval = self._get_sample_range(index)
        assert start == index and interval == 1
        assert end == start + self.training_frame_num
        assert not self.test_mode

        ret = None
        for i in range(start, end, interval):
            data_i = self.prepare_train_data_single(i)
            if data_i is None:
                return None

            if ret is None:
                ret = {key: [] for key in data_i.keys()}

            for key, value in data_i.items():
                ret[key].append(value)

        if self.do_pred:
            if True:
                pred_data = self.prepare_pred(start, end, interval, index)

            ret.update(pred_data)

            mapping = ret['mapping']
            if not self.test_mode:
                mapping['training_frame_num'] = self.training_frame_num

        ret = self.pipeline_post(ret)

        return ret

    def prepare_pred(self, start, end, interval, index):
        results = {}
        if not hasattr(self, 'nuscenes'):
            self.prepare_nuscenes()

        same_scene = self.is_the_same_scene(start, end, 12)

        mapping = {}

        self.get_pred_agents(results, start, end, interval, mapping)
        self.get_pred_lanes(results, start, end, interval, mapping)

        info = self.data_infos[end - 1]
        sample = self.helper.data.get('sample', info['token'])
        scene_id = self.helper.data.get('scene', sample['scene_token'])['name']

        mapping.update(dict(
            timestamp=info['timestamp'] / 1e6,
            timestamp_origin=info['timestamp'],
            sample_token=info['token'],
            scene_id=self.helper.data.get('scene', sample['scene_token'])['name'],
            same_scene=same_scene,
            index=index,
        ))

        if hasattr(self, 'work_dir'):
            mapping['work_dir'] = self.work_dir

        results['mapping'] = mapping
        return results

    def prepare_test_data_single(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline_single(input_dict)
        return example

    def prepare_test_data(self, index):
        start, end, interval = self._get_sample_range(index)
        assert start == end - 1

        ret = None
        for i in range(start, end, interval):
            data_i = self.prepare_test_data_single(i)

            if ret is None:
                ret = {key: [] for key in data_i.keys()}

            for key, value in data_i.items():
                ret[key].append(value)

        if self.generate_nuscenes_prediction_infos_val:
            data_i = self.prepare_test_data_single(index)
            self.generate_prediction(data_i, self.data_infos[start], index)

        if self.do_pred:
            pred_data = self.prepare_pred(end - 3, end, interval, index)
            ret.update(pred_data)

        ret = self.pipeline_post(ret)
        return ret

    def _rand_another(self, idx):
        """Randomly get another item with the same flag.

        Returns:
            int: Another index of item with the same flag.
        """
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:

            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue

            return data

    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            sample_token = self.data_infos[sample_id]['token']
            if det is None:
                nusc_annos[sample_token] = annos
                continue
            boxes = output_to_nusc_box_tracking(det)
            boxes = lidar_nusc_box_to_global(self.data_infos[sample_id], boxes,
                                             mapped_class_names,
                                             self.eval_detection_configs,
                                             self.eval_version,
                                             include_prediction=True)
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in [
                        'car',
                        'construction_vehicle',
                        'bus',
                        'truck',
                        'trailer',
                    ]:
                        attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'
                    else:
                        attr = NuScenesTrackDatasetRadar.DefaultAttribute[name]
                else:
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    elif name in ['bus']:
                        attr = 'vehicle.stopped'
                    else:
                        attr = NuScenesTrackDatasetRadar.DefaultAttribute[name]

                if True:
                    nusc_anno = dict(
                        sample_token=sample_token,
                        translation=box.center.tolist(),
                        size=box.wlh.tolist(),
                        rotation=box.orientation.elements.tolist(),
                        velocity=box.velocity[:2].tolist(),
                        tracking_name=name,
                        attribute_name=attr,
                        tracking_score=box.score,
                        tracking_id=box.token,
                        pred_outputs=box.pred_outputs if hasattr(box, 'pred_outputs') else None,
                        pred_probs=box.pred_probs if hasattr(box, 'pred_probs') else None,
                    )
                # print(nusc_anno)
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        if True:
            res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        # evaluate = format_results + _evaluate_single
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        output_dir = osp.join(*osp.split(result_path)[:-1])

        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        from nuscenes.eval.tracking.evaluate import TrackingEval
        from nuscenes.eval.common.config import config_factory as track_configs

        cfg = track_configs("tracking_nips_2019")
        if True:
            nusc_eval = TrackingEval(
                config=cfg,
                result_path=result_path,
                eval_set=eval_set_map[self.version],
                output_dir=output_dir,
                verbose=True,
                nusc_version=self.version,
                nusc_dataroot=self.data_root
            )
        metrics = nusc_eval.main()

        # record metrics
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        print(metrics)
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        keys = ['amota', 'amotp', 'recall', 'motar',
                'gt', 'mota', 'motp', 'mt', 'ml', 'faf',
                'tp', 'fp', 'fn', 'ids', 'frag', 'tid', 'lgd']
        for key in keys:
            detail['{}/{}'.format(metric_prefix, key)] = metrics[key]
        return detail

    def format_results(self, results, jsonfile_prefix=None):
        # store at osp.join(jsonfile_prefix, 'results_nusc.json')
        # if jsonfile_prefix is None, use temp file
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a \
                dict containing the json filepaths, `tmp_dir` is the temporal \
                directory created for saving json files when \
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        result_files = self._format_bbox(results, jsonfile_prefix)

        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 out_dir=None,
                 pipeline=None):
        # evaluate = format_results + _evaluate_single
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        assert isinstance(result_files, str)
        results_dict = self._evaluate_single(result_files)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        return results_dict

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                file_client_args=dict(backend='disk')),
            dict(
                type='DefaultFormatBundle3D',
                class_names=self.CLASSES,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ]
        return Compose(pipeline)

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)


class NuScenesTrackingBox(NuScenesBox):
    def __init__(self,
                 center: List[float],
                 size: List[float],
                 orientation: Quaternion,
                 label: int = np.nan,
                 score: float = np.nan,
                 velocity: Tuple = (np.nan, np.nan, np.nan),
                 name: str = None,
                 token: str = None,
                 pred_outputs: np.ndarray = None,
                 pred_probs: np.ndarray = None,
                 ):
        """
        :param center: Center of box given as x, y, z.
        :param size: Size of box in width, length, height.
        :param orientation: Box orientation.
        :param label: Integer label, optional.
        :param score: Classification score, optional.
        :param velocity: Box velocity in x, y, z direction.
        :param name: Box name, optional. Can be used e.g. for denote category name.
        :param token: Unique string identifier from DB.
        """
        super(NuScenesTrackingBox, self).__init__(center, size, orientation, label,
                                                  score, velocity, name, token)
        self.pred_outputs = pred_outputs
        self.pred_probs = pred_probs

    def rotate(self, quaternion: Quaternion) -> None:
        self.center = np.dot(quaternion.rotation_matrix, self.center)
        self.orientation = quaternion * self.orientation
        self.velocity = np.dot(quaternion.rotation_matrix, self.velocity)

    def copy(self) -> 'NuScenesTrackingBox':
        return copy.deepcopy(self)


def output_to_nusc_box_tracking(detection):
    """Convert the output to the box class in the nuScenes.
    Args:
        detection (dict): Detection results.
            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

        tracking (bool): if convert for tracking evaluation
    Returns:
        list[:obj:`NuScenesBox`]: List of NuScenesTrackingBoxes.
    """
    box3d = detection['boxes_3d']

    # overwrite the scores with the tracking scores
    if 'track_scores' in detection.keys() and detection['track_scores'] is not None:
        scores = detection['track_scores'].numpy()
    else:
        scores = detection['scores_3d'].numpy()

    labels = detection['labels_3d'].numpy()
    pred_outputs_in_ego = detection.get('pred_outputs_in_ego', None)
    pred_probs_in_ego = detection.get('pred_probs_in_ego', None)

    if 'track_ids' in detection.keys() and detection['track_ids'] is not None:
        track_ids = detection['track_ids']
    else:
        track_ids = [None for _ in range(len(box3d))]

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()
    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    box_yaw = -box_yaw - np.pi / 2

    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        velocity = (*box3d.tensor[i, 7:9], 0.0)
        # velo_val = np.linalg.norm(box3d[i, 7:9])
        # velo_ori = box3d[i, 6]
        # velocity = (
        # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
        box = NuScenesTrackingBox(
            box_gravity_center[i],
            box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity,
            token=str(track_ids[i]),
            pred_outputs=pred_outputs_in_ego[i] if pred_outputs_in_ego is not None else None,
            pred_probs=pred_probs_in_ego[i] if pred_probs_in_ego is not None else None,
        )
        box_list.append(box)
    return box_list


def lidar_nusc_box_to_global(info,
                             boxes,
                             classes,
                             eval_configs,
                             eval_version='detection_cvpr_2019',
                             include_prediction=False,
                             ):
    """Convert the box from ego to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.
        eval_version (str): Evaluation version.
            Default: 'detection_cvpr_2019'

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(pyquaternion.Quaternion(info['lidar2ego_rotation']))
        box.translate(np.array(info['lidar2ego_translation']))
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info['ego2global_rotation']))
        box.translate(np.array(info['ego2global_translation']))
        box_list.append(box)

        if include_prediction:
            assert hasattr(box, 'pred_outputs')
            if box.pred_outputs is not None:
                # Move pred outputs from ego to global
                assert box.pred_outputs.shape == (6, 12, 2)
                for i in range(box.pred_outputs.shape[0]):
                    for j in range(box.pred_outputs.shape[1]):
                        box.pred_outputs[i, j] = utils.get_transform_and_rotate(box.pred_outputs[i, j], np.array(info['ego2global_translation']), info['ego2global_rotation'])
            else:
                box.pred_outputs = np.zeros((6, 12, 2))
                translation = np.array([box.center[0], box.center[1]])
                box.pred_outputs[:] = translation[np.newaxis, np.newaxis, :]

    return box_list


def get_lanes_in_radius(x: float, y: float, radius: float,
                        discretization_meters: float,
                        map_api: NuScenesMap) -> Dict[str, List[Tuple[float, float, float]]]:
    """
    Retrieves all the lanes and lane connectors in a radius of the query point.
    :param x: x-coordinate of point in global coordinates.
    :param y: y-coordinate of point in global coordinates.
    :param radius: Any lanes within radius meters of the (x, y) point will be returned.
    :param discretization_meters: How finely to discretize the lane. If 1 is given, for example,
        the lane will be discretized into a list of points such that the distances between points
        is approximately 1 meter.
    :param map_api: The NuScenesMap instance to query.
    :return: Mapping from lane id to list of coordinate tuples in global coordinate system.
    """

    lanes = map_api.get_records_in_radius(x, y, radius, ['lane', 'lane_connector'])
    lanes = lanes['lane'] + lanes['lane_connector']
    lanes = map_api.discretize_lanes(lanes, discretization_meters)

    return lanes


def load_all_maps(helper: PredictHelper, verbose: bool = False) -> Dict[str, NuScenesMap]:
    """
    Loads all NuScenesMap instances for all available maps.
    :param helper: Instance of PredictHelper.
    :param verbose: Whether to print to stdout.
    :return: Mapping from map-name to the NuScenesMap api instance.
    """
    dataroot = helper.data.dataroot
    maps = {}

    for map_name in locations:
        if verbose:
            print(f'static_layers.py - Loading Map: {map_name}')

        maps[map_name] = NuScenesMap(dataroot, map_name=map_name)

    return maps
