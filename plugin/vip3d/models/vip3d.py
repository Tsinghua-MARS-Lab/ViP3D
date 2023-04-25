import json
import os.path
import pickle
import random
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet.models import build_loss
from pyquaternion import Quaternion

from mmdet3d.core.bbox.coders import build_bbox_coder
from ...mmdet3d_plugin.core.bbox.util import normalize_bbox, denormalize_bbox
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from ...mmdet3d_plugin.models.utils.grid_mask import GridMask
from .attention_dert3d import inverse_sigmoid
from . import predictor_lib
from .memory_bank import build_memory_bank
from .qim import build_qim
from .radar_encoder import build_radar_encoder
from .. import utils as predictor_utils
from ..structures import Instances


class RuntimeTrackerBase(object):
    def __init__(self, score_thresh=None, filter_score_thresh=None, miss_tolerance=5):
        self.score_thresh = score_thresh
        self.filter_score_thresh = filter_score_thresh
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0
        self.link_track_id = False
        self.last_track_instances = None

    def clear(self):
        self.max_obj_id = 0

    def update(self, track_instances: Instances):
        track_instances.disappear_time[track_instances.scores >= self.score_thresh] = 0
        for i in range(len(track_instances)):
            if track_instances.obj_idxes[i] == -1 and track_instances.scores[i] >= self.score_thresh:
                # new track
                # print("track {} has score {}, assign obj_id {}".format(i, track_instances.scores[i], self.max_obj_id))

                if True:
                    track_instances.obj_idxes[i] = self.max_obj_id
                    self.max_obj_id += 1
            elif track_instances.obj_idxes[i] >= 0 and track_instances.scores[i] < self.filter_score_thresh:
                # sleep time ++
                track_instances.disappear_time[i] += 1
                if track_instances.disappear_time[i] >= self.miss_tolerance:
                    # mark deaded tracklets: Set the obj_id to -1.
                    # TODO: remove it by following functions
                    # Then this track will be removed by TrackEmbeddingLayer.
                    track_instances.obj_idxes[i] = -1

    def update_fix_label(self, track_instances: Instances, old_class_scores):
        track_instances.disappear_time[track_instances.scores >= self.score_thresh] = 0
        for i in range(len(track_instances)):
            if track_instances.obj_idxes[i] == -1 and track_instances.scores[i] >= self.score_thresh:
                # new track
                # print("track {} has score {}, assign obj_id {}".format(i, track_instances.scores[i], self.max_obj_id))
                track_instances.obj_idxes[i] = self.max_obj_id
                self.max_obj_id += 1
            elif track_instances.obj_idxes[i] >= 0 and track_instances.scores[i] < self.filter_score_thresh:
                # sleep time ++
                track_instances.disappear_time[i] += 1
                # keep class unchanged!
                track_instances.pred_logits[i] = old_class_scores[i]
                if track_instances.disappear_time[i] >= self.miss_tolerance:
                    # mark deaded tracklets: Set the obj_id to -1.
                    # TODO: remove it by following functions
                    # Then this track will be removed by TrackEmbeddingLayer.
                    track_instances.obj_idxes[i] = -1
            elif track_instances.obj_idxes[i] >= 0 and track_instances.scores[i] >= self.filter_score_thresh:
                # keep class unchanged!
                track_instances.pred_logits[i] = old_class_scores[i]


@DETECTORS.register_module()
class ViP3D(MVXTwoStageDetector):
    def __init__(self,
                 embed_dims=256,
                 num_query=300,
                 num_classes=7,
                 bbox_coder=None,
                 qim_args=None,
                 mem_cfg=None,
                 radar_encoder=None,
                 fix_feats=False,
                 score_thresh=None,
                 filter_score_thresh=None,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 loss_cfg=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 do_pred=False,
                 predictor=None,
                 relative_pred=False,
                 agents_layer_0=False,
                 agents_layer_0_num=2,
                 only_matched_query=False,
                 add_branch=False,
                 add_branch_2=False,
                 ):
        super(ViP3D,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.num_classes = num_classes
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range

        self.embed_dims = embed_dims
        self.num_query = num_query
        self.fix_feats = fix_feats
        if self.fix_feats:
            self.img_backbone.eval()
            self.img_neck.eval()
        self.bbox_size_fc = nn.Linear(self.embed_dims, 3)

        if True:
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)
            self.reference_points = nn.Linear(self.embed_dims, 3)

        self.track_base = RuntimeTrackerBase(
            score_thresh=score_thresh,
            filter_score_thresh=filter_score_thresh,
            miss_tolerance=5)

        self.query_interact = build_qim(
            qim_args,
            dim_in=embed_dims,
            hidden_dim=embed_dims,
            dim_out=embed_dims,
        )

        self.memory_bank = build_memory_bank(
            args=mem_cfg,
            dim_in=embed_dims,
            hidden_dim=embed_dims,
            dim_out=embed_dims,
        )
        self.mem_bank_len = 0 if self.memory_bank is None else self.memory_bank.max_his_length
        self.criterion = build_loss(loss_cfg)
        self.test_track_instances = None
        self.l2g_r_mat = None
        self.l2g_t = None

        self.radar_encoder = build_radar_encoder(radar_encoder)

        self.do_pred = do_pred
        self.relative_pred = relative_pred

        self.add_branch = add_branch
        self.add_branch_2 = add_branch_2
        if self.add_branch:
            from mmcv.utils import build_from_cfg
            from mmcv.cnn.bricks.registry import (TRANSFORMER_LAYER)
            self.add_branch_mlp = predictor_lib.MLP(256, 256)
            self.add_branch_attention = build_from_cfg(predictor_utils.get_attention_cfg(), TRANSFORMER_LAYER)

        if True:
            self.agents_layer_0 = agents_layer_0
            if self.agents_layer_0:
                self.agents_layer_mlp_0 = nn.Sequential(*[predictor_lib.MLP(256, 256) for _ in range(agents_layer_0_num)])

        self.only_matched_query = only_matched_query

        if self.do_pred:
            from .predictor_vectornet import VectorNet
            self.predictor = VectorNet(**predictor)
            self.empty_linear = nn.Linear(embed_dims, embed_dims)

    def velo_update(self, ref_pts, velocity, l2g_r1, l2g_t1, l2g_r2, l2g_t2,
                    time_delta):
        '''
        Args:
            ref_pts (Tensor): (num_query, 3).  in inevrse sigmoid space
            velocity (Tensor): (num_query, 2). m/s
                in lidar frame. vx, vy
            global2lidar (np.Array) [4,4].
        Outs:
            ref_pts (Tensor): (num_query, 3).  in inevrse sigmoid space
        '''
        # print(l2g_r1.type(), l2g_t1.type(), ref_pts.type())

        if time_delta > 1. or time_delta < 0.:
            time_delta = torch.tensor(0., device=time_delta.device, dtype=time_delta.dtype)

        # class 'torch.Tensor'> tensor([0.5005], device='cuda:0', dtype=torch.float64)
        time_delta = time_delta.type(torch.float)
        num_query = ref_pts.size(0)
        velo_pad_ = velocity.new_zeros((num_query, 1))
        velo_pad = torch.cat((velocity, velo_pad_), dim=-1)

        reference_points = ref_pts.sigmoid().clone()
        pc_range = self.pc_range
        reference_points[..., 0:1] = reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]

        # Update first: according to query velocity
        reference_points = reference_points + velo_pad * time_delta

        ref_pts = reference_points @ l2g_r1.T + l2g_t1
        ref_pts = (ref_pts - l2g_t2) @ torch.linalg.inv(l2g_r2).T.type(torch.float)

        ref_pts[..., 0:1] = (ref_pts[..., 0:1] - pc_range[0]) / (pc_range[3] - pc_range[0])
        ref_pts[..., 1:2] = (ref_pts[..., 1:2] - pc_range[1]) / (pc_range[4] - pc_range[1])
        ref_pts[..., 2:3] = (ref_pts[..., 2:3] - pc_range[2]) / (pc_range[5] - pc_range[2])

        ref_pts = inverse_sigmoid(ref_pts)

        return ref_pts

    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        voxels, num_points, coors = self.voxelize(pts)

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        B = img.size(0)
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)

            if self.use_grid_mask:
                # img: fp16
                img = self.grid_mask(img)
                # img: fp32
            # slightly different

            # img: fp32
            img_feats = self.img_backbone(img)
            # img: fp16

        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, points, img, radar=None, img_metas=None):
        """Extract features from images and points."""
        if radar is not None:
            radar_feats = self.radar_encoder(radar)
        else:
            radar_feats = None
        if self.fix_feats:
            with torch.no_grad():
                img_feats = self.extract_img_feat(img, img_metas)
        else:
            img_feats = self.extract_img_feat(img, img_metas)
        return (img_feats, radar_feats, None)

    def _targets_to_instances(self, gt_bboxes_3d=None,
                              gt_labels_3d=None, instance_inds=None,
                              img_shape=(1, 1,)):
        gt_instances = Instances(tuple(img_shape))
        gt_instances.boxes = gt_bboxes_3d
        gt_instances.labels = gt_labels_3d
        gt_instances.obj_ids = instance_inds
        return gt_instances

    def _generate_empty_tracks(self, proposals=None):
        track_instances = Instances((1, 1))
        num_queries, dim = self.query_embedding.weight.shape  # (300, 256 * 2)
        device = self.query_embedding.weight.device

        # query is learnable embedding
        # ref_pts is decoded from query by nn.Linear(self.embed_dims, 3)
        query = self.query_embedding.weight

        # init boxes: xy, wl, z, h, sin, cos, vx, vy, vz
        box_sizes = self.bbox_size_fc(query[..., :dim // 2])
        pred_boxes_init = torch.zeros(
            (len(query), 10), dtype=torch.float, device=device)

        pred_boxes_init[..., 2:4] = box_sizes[..., 0:2]
        pred_boxes_init[..., 5:6] = box_sizes[..., 2:3]

        if True:
            track_instances.ref_pts = self.reference_points(query[..., :dim // 2])
            track_instances.query = query

        track_instances.output_embedding = torch.zeros(
            (len(track_instances), dim >> 1), device=device)

        track_instances.obj_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long, device=device)
        track_instances.matched_gt_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long, device=device)
        track_instances.disappear_time = torch.zeros(
            (len(track_instances),), dtype=torch.long, device=device)

        track_instances.scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device)
        track_instances.track_scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device)
        # xy, wl, z, h, sin, cos, vx, vy, vz
        track_instances.pred_boxes = pred_boxes_init

        track_instances.pred_logits = torch.zeros(
            (len(track_instances), self.num_classes),
            dtype=torch.float, device=device)

        mem_bank_len = self.mem_bank_len
        track_instances.mem_bank = torch.zeros(
            (len(track_instances), mem_bank_len, dim // 2),
            dtype=torch.float32, device=device)
        track_instances.mem_padding_mask = torch.ones(
            (len(track_instances), mem_bank_len),
            dtype=torch.bool, device=device)
        track_instances.save_period = torch.zeros(
            (len(track_instances),), dtype=torch.float32, device=device)

        return track_instances.to(device)

    def _copy_tracks_for_loss(self, tgt_instances):

        device = self.query_embedding.weight.device
        track_instances = Instances((1, 1))

        track_instances.obj_idxes = deepcopy(tgt_instances.obj_idxes)
        track_instances.matched_gt_idxes = deepcopy(tgt_instances.matched_gt_idxes)
        track_instances.disappear_time = deepcopy(tgt_instances.disappear_time)

        track_instances.scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device)
        track_instances.track_scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device)
        track_instances.pred_boxes = torch.zeros(
            (len(track_instances), 10), dtype=torch.float, device=device)
        track_instances.pred_logits = torch.zeros(
            (len(track_instances), self.num_classes),
            dtype=torch.float, device=device)

        track_instances.save_period = deepcopy(tgt_instances.save_period)
        return track_instances.to(self.query_embedding.weight.device)

    @force_fp32(apply_to=('img', 'points'))
    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.

        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            if self.do_pred:
                self.predictor.decoder.do_eval = True
            return self.forward_test(**kwargs)

    # @auto_fp16(apply_to=('img', 'radar'))
    def _forward_single(self, points, img, radar, img_metas, track_instances,
                        l2g_r1=None, l2g_t1=None, l2g_r2=None, l2g_t2=None,
                        time_delta=None, is_last_frame=False,
                        gt_bboxes_3d=None,
                        gt_labels_3d=None,
                        mapping=None,
                        ):
        '''
        Warnning: Only Support BS=1
        img: shape [B, num_cam, 3, H, W]

        if l2g_r2 is None or l2g_t2 is None:
            no need to call velo update
        '''
        # l2g_r2 is rotation matrix of next frame

        B, num_cam, _, H, W = img.shape

        if True:
            img_feats, radar_feats, pts_feats = self.extract_feat(
                points, img=img, radar=radar, img_metas=img_metas)
            # img_feats = [a.clone() for a in img_feats]

            # output_classes: [num_dec, B, num_query, num_classes]
            # query_feats: [B, num_query, embed_dim]

            ref_box_sizes = torch.cat(
                [track_instances.pred_boxes[:, 2:4],
                 track_instances.pred_boxes[:, 5:6]], dim=1)

            output_classes, output_coords, \
                query_feats, last_ref_pts = self.pts_bbox_head(
                img_feats, radar_feats, track_instances.query,
                track_instances.ref_pts, ref_box_sizes, img_metas, )

        if self.add_branch:
            self.update_history_img_list(img_metas, img, img_feats)

        out = {'pred_logits': output_classes[-1],
               'pred_boxes': output_coords[-1],
               'ref_pts': last_ref_pts}

        with torch.no_grad():
            track_scores = output_classes[-1, 0, :].sigmoid().max(dim=-1).values

        # Step-1 Update track instances with current prediction
        # [nb_dec, bs, num_query, xxx]
        nb_dec = output_classes.size(0)

        # the track id will be assigned by the mather.
        # only copy matched_gt_idxes, obj_idxes, etc.
        track_instances_list = [self._copy_tracks_for_loss(track_instances) for i in range(nb_dec - 1)]
        track_instances.output_embedding = query_feats[0]  # [300, feat_dim]
        velo = output_coords[-1, 0, :, -2:]  # [num_query, 3]

        if True:
            if l2g_r2 is not None:
                ref_pts = self.velo_update(
                    last_ref_pts[0], velo, l2g_r1, l2g_t1, l2g_r2, l2g_t2,
                    time_delta=time_delta)
            else:
                ref_pts = last_ref_pts[0]
            track_instances.ref_pts = ref_pts

        # track_instances.query = torch.cat((track_instances.query[:, :self.embed_dims // 2],
        #   query_feats[0]), dim=1)

        track_instances_list.append(track_instances)
        for i in range(nb_dec):
            track_instances = track_instances_list[i]
            track_instances.scores = track_scores
            track_instances.pred_logits = output_classes[i, 0]  # [300, num_cls]
            track_instances.pred_boxes = output_coords[i, 0]  # [300, box_dim]

            # used keys in 'track_instances': 'pred_logits', 'pred_boxes', 'obj_idxes', 'matched_gt_idxes',
            # modified keys: 'matched_gt_idxes', 'obj_idxes'
            track_instances = self.criterion.match_for_single_frame(
                track_instances, i, if_step=(i == (nb_dec - 1)))

        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)

        tmp = {}
        tmp['track_instances'] = track_instances
        out_track_instances = self.query_interact(tmp)  # see qim.py
        return out_track_instances

    def forward_train(self,
                      points=None,
                      img=None,
                      radar=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      instance_inds=None,
                      l2g_r_mat=None,
                      l2g_t=None,
                      gt_bboxes_ignore=None,
                      timestamp=None,
                      instance_idx_2_labels=None,
                      **kwargs,
                      ):
        """Forward training function.
        Args:
            points (list(list[torch.Tensor]), optional): B-T-sample
                Points of each sample.
                Defaults to None.
            img (Torch.Tensor) of shape [B, T, num_cam, 3, H, W]
            radar (Torch.Tensor) of shape [B, T, num_points, radar_dim]
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            lidar2img = img_metas[bs]['lidar2img'] of shape [3, 6, 4, 4]. list
                of list of list of 4x4 array
            gt_bboxes_3d (list[list[:obj:`BaseInstance3DBoxes`]], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[list[torch.Tensor]], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            l2g_r_mat (list[Tensor]). element shape [T, 3, 3]
            l2g_t (list[Tensor]). element shape [T, 3]
                points @ R_Mat + T
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """

        # [T, 3, 3]
        l2g_r_mat = l2g_r_mat[0]
        # change to [T, 1, 3]
        l2g_t = l2g_t[0].unsqueeze(dim=1)

        timestamp = timestamp

        bs = img.size(0)
        num_frame = img.size(1)
        track_instances = self._generate_empty_tracks()

        # init gt instances!
        gt_instances_list = []
        for i in range(num_frame):
            gt_instances = Instances((1, 1))
            boxes = gt_bboxes_3d[0][i].tensor.to(img.device)
            # normalize gt bboxes here!
            boxes = normalize_bbox(boxes, self.pc_range)

            gt_instances.boxes = boxes
            gt_instances.labels = gt_labels_3d[0][i]
            gt_instances.obj_ids = instance_inds[0][i]
            gt_instances_list.append(gt_instances)

        self.criterion.initialize_for_single_clip(gt_instances_list)

        others_dict = {}
        mapping = None
        if self.do_pred:
            mapping = kwargs['mapping'][0]
            same_scene = mapping['same_scene']
            valid_pred = mapping['valid_pred']

        if True:
            # for bs 1
            lidar2img = img_metas[0]['lidar2img']  # [T, num_cam]
            for i in range(num_frame):
                points_single = [p_[i] for p_ in points]
                img_single = torch.stack([img_[i] for img_ in img], dim=0)
                radar_single = torch.stack([radar_[i] for radar_ in radar], dim=0)

                img_metas_single = deepcopy(img_metas)
                img_metas_single[0]['lidar2img'] = lidar2img[i]

                if i == num_frame - 1:
                    l2g_r2 = None
                    l2g_t2 = None
                    time_delta = None
                else:
                    l2g_r2 = l2g_r_mat[i + 1]
                    l2g_t2 = l2g_t[i + 1]
                    time_delta = timestamp[i + 1] - timestamp[i]

                is_last_frame = i == num_frame - 1

                if True:
                    track_instances = self._forward_single(points_single, img_single,
                                                           radar_single, img_metas_single,
                                                           track_instances,
                                                           l2g_r_mat[i], l2g_t[i],
                                                           l2g_r2, l2g_t2, time_delta, is_last_frame,
                                                           gt_bboxes_3d=predictor_utils.tensors_tracking_to_detection(gt_bboxes_3d, i),
                                                           gt_labels_3d=predictor_utils.tensors_tracking_to_detection(gt_labels_3d, i),
                                                           mapping=mapping)

                if True:
                    track_instances = Instances.cat([track_instances, self._generate_empty_tracks()])

                if self.add_branch:
                    track_ids = []
                    decoded_boxes = []

                    all_decoded_boxes = predictor_utils.to_numpy(predictor_utils.get_decoded_boxes(track_instances.pred_boxes, self.pc_range, img_metas).tensor)
                    for j in range(len(track_instances)):
                        obj_id = track_instances.obj_idxes[j].item()
                        if obj_id != -1 and obj_id != -2:
                            track_ids.append(obj_id)

                            # clip
                            all_decoded_boxes[j][3:6] = np.clip(all_decoded_boxes[j][3:6], -20, 20)
                            decoded_boxes.append(all_decoded_boxes[j])

                    if i == 0:
                        self.track_idx_2_boxes = defaultdict(dict)
                        self.track_idx_2_boxes_in_lidar = defaultdict(dict)

                    # mapping must use last mapping
                    r_index_2_rotation_and_transform = mapping['r_index_2_rotation_and_transform']
                    if valid_pred:
                        predictor_utils.update_track_idx_2_boxes(self.track_idx_2_boxes, track_ids, decoded_boxes, r_index_2_rotation_and_transform[i], i)
                        predictor_utils.update_track_idx_2_boxes_in_lidar(self.track_idx_2_boxes_in_lidar, track_ids, decoded_boxes, r_index_2_rotation_and_transform[i], i)

        if self.do_pred:

            instance_idx_2_labels = instance_idx_2_labels[0]
            device = track_instances.output_embedding.device

            if not valid_pred:
                loss = None
            else:
                # valid agents which have been matched with GT
                agents_indices = []
                labels_list = []
                labels_is_valid_list = []
                if self.add_branch:
                    track_scores = []
                    track_ids = []
                    track_labels = []
                    all_track_labels = predictor_utils.get_labels_from_pred_logits(track_instances.pred_logits, self.bbox_coder.num_classes)

                for j in range(len(track_instances)):
                    # obj_idxes is obtained by matching
                    obj_id = track_instances.obj_idxes[j].item()
                    if obj_id != -1 and obj_id != -2:
                        def run(future_traj=None,
                                future_traj_relative=None,
                                future_traj_is_valid=None,
                                past_traj=None,
                                past_traj_is_valid=None,
                                category=None,
                                past_boxes=None,
                                **kwargs):
                            if self.relative_pred:
                                labels_list.append(future_traj_relative)
                            else:
                                labels_list.append(future_traj)
                            labels_is_valid_list.append(future_traj_is_valid)

                        run(**instance_idx_2_labels[obj_id])
                        agents_indices.append(j)

                        if self.add_branch:
                            track_scores.append(track_instances.scores[j].item())
                            track_ids.append(obj_id)
                            track_labels.append(all_track_labels[j])

                if len(agents_indices) > 0:
                    if True:
                        if True:
                            output_embedding = track_instances.output_embedding

                            if self.add_branch:
                                output_embedding = output_embedding[torch.tensor(agents_indices, dtype=torch.long, device=device)]
                                _, _, past_boxes_list_in_lidar, _, _ = \
                                    predictor_utils.extract_from_track_idx_2_boxes(self.track_idx_2_boxes_in_lidar, track_scores, track_ids, track_labels, mapping, num_frame - 1)
                                query = self.add_branch_update_query(output_embedding, past_boxes_list_in_lidar[:, -1, :3], device)
                                output_embedding = output_embedding + query

                            output_embedding = self.output_embedding_forward(output_embedding)

                        if self.add_branch:
                            loss, outputs, _ = self.predictor(agents=output_embedding.unsqueeze(0),
                                                              device=device,
                                                              labels=[np.array(labels_list)],
                                                              labels_is_valid=[np.array(labels_is_valid_list)],
                                                              # agents_indices=np.array(agents_indices, dtype=int),
                                                              **kwargs)
                        elif self.only_matched_query:
                            output_embedding = output_embedding[torch.tensor(agents_indices, dtype=torch.long, device=device)]
                            loss, outputs, _ = self.predictor(agents=output_embedding.unsqueeze(0),
                                                              device=device,
                                                              labels=[np.array(labels_list)],
                                                              labels_is_valid=[np.array(labels_is_valid_list)],
                                                              # agents_indices=np.array(agents_indices, dtype=int),
                                                              **kwargs)
                        else:
                            loss, outputs, _ = self.predictor(agents=output_embedding.unsqueeze(0),
                                                              device=device,
                                                              labels=[np.array(labels_list)],
                                                              labels_is_valid=[np.array(labels_is_valid_list)],
                                                              agents_indices=np.array(agents_indices, dtype=int),
                                                              **kwargs)
                else:
                    loss = None

            if loss is None:
                # zero loss as a placeholder
                loss = torch.abs(self.empty_linear(torch.zeros(self.embed_dims, device=device, dtype=torch.float))).mean()

            self.criterion.update_prediction_loss(loss)

        outputs = self.criterion.losses_dict
        return outputs, others_dict

    def _inference_single(self, points, img, radar, img_metas, track_instances,
                          l2g_r1=None, l2g_t1=None, l2g_r2=None, l2g_t2=None,
                          time_delta=None,
                          gt_bboxes_3d=None,
                          gt_labels_3d=None):
        '''
        Warnning: Only Support BS=1
        img: shape [B, num_cam, 3, H, W]
        '''

        # velo update:
        active_inst = track_instances[track_instances.obj_idxes >= 0]
        other_inst = track_instances[track_instances.obj_idxes < 0]

        if l2g_r2 is not None and len(active_inst) > 0 and l2g_r1 is not None:
            ref_pts = active_inst.ref_pts
            velo = active_inst.pred_boxes[:, -2:]
            ref_pts = self.velo_update(
                ref_pts, velo, l2g_r1, l2g_t1, l2g_r2, l2g_t2,
                time_delta=time_delta)
            active_inst.ref_pts = ref_pts
        track_instances = Instances.cat([other_inst, active_inst])

        try:
            B, num_cam, _, H, W = img.shape
        except Exception:
            assert False
        B, num_cam, _, H, W = img.shape

        """
        # what does it do? img_feats = [a.clone() for a in img_feats]
        """
        if True:
            img_feats, radar_feats, pts_feats = self.extract_feat(
                points, img=img, radar=radar, img_metas=img_metas)
            img_feats = [a.clone() for a in img_feats]

            # output_classes: [num_dec, B, num_query, num_classes]
            # query_feats: [B, num_query, embed_dim]
            ref_box_sizes = torch.cat(
                [track_instances.pred_boxes[:, 2:4],
                 track_instances.pred_boxes[:, 5:6]], dim=1)

            output_classes, output_coords, \
                query_feats, last_ref_pts = self.pts_bbox_head(
                img_feats, radar_feats, track_instances.query,
                track_instances.ref_pts, ref_box_sizes, img_metas, )

        if self.add_branch:
            self.update_history_img_list(img_metas, img, img_feats)

        out = {'pred_logits': output_classes[-1],
               'pred_boxes': output_coords[-1],
               'ref_pts': last_ref_pts}

        # TODO: Why no max?
        track_scores = output_classes[-1, 0, :].sigmoid().max(dim=-1).values
        # track_scores = output_classes[-1, 0, :, 0].sigmoid()

        # Step-1 Update track instances with current prediction
        # [nb_dec, bs, num_query, xxx]

        # each track will be assigned an unique global id by the track base.
        track_instances.scores = track_scores
        # track_instances.track_scores = track_scores  # [300]
        track_instances.pred_logits = output_classes[-1, 0]  # [300, num_cls]
        track_instances.pred_boxes = output_coords[-1, 0]  # [300, box_dim]
        track_instances.output_embedding = query_feats[0]  # [300, feat_dim]

        track_instances.ref_pts = last_ref_pts[0]

        self.track_base.update(track_instances)
        # self.track_base.update_fix_label(track_instances, old_class_scores)

        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)
            # track_instances.track_scores = track_instances.track_scores[..., 0]
            # track_instances.scores = track_instances.track_scores.sigmoid()

        # Step-2 Update track instances using matcher

        tmp = {}
        tmp['track_instances'] = track_instances
        out_track_instances = self.query_interact(tmp)
        return out_track_instances

    def forward_test(self,
                     points=None,
                     img=None,
                     radar=None,
                     img_metas=None,
                     timestamp=1e6,
                     l2g_r_mat=None,
                     l2g_t=None,
                     instance_idx_2_labels=None,
                     mapping=None,
                     gt_bboxes_3d=None,
                     gt_labels_3d=None,
                     **kwargs,
                     ):
        """Forward test function.
        only support bs=1, single-gpu, num_frame=1 test
        Args:
            points (list(list[torch.Tensor]), optional): B-T-sample
                Points of each sample.
                Defaults to None.
            img (Torch.Tensor) of shape [B, T, num_cam, 3, H, W]
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            lidar2img = img_metas[bs]['lidar2img'] of shape [3, 6, 4, 4]. list
                of list of list of 4x4 array
            gt_bboxes_3d (list[list[:obj:`BaseInstance3DBoxes`]], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[list[torch.Tensor]], optional): Ground truth labels
                of 3D boxes. Defaults to None.

            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        # [3, 3]
        l2g_r_mat = l2g_r_mat[0][0]
        # change to [1, 3]
        l2g_t = l2g_t[0].unsqueeze(dim=1)[0]

        bs = img.size(0)
        num_frame = img.size(1)

        timestamp = timestamp[0]
        device = img.device

        # track_instances of last frame
        if self.test_track_instances is None:
            if True:
                track_instances = self._generate_empty_tracks()
                self.test_track_instances = track_instances
            self.timestamp = timestamp[0]

        new_query = False
        no_metric_now = False
        self.track_base.link_track_id = False

        # TODO: use scene tokens?
        if True:
            assert timestamp[0] - self.timestamp >= 0
        if True:
            if timestamp[0] - self.timestamp > 10 or new_query:
                if True:
                    track_instances = self._generate_empty_tracks()
                time_delta = None
                l2g_r1 = None
                l2g_t1 = None
                l2g_r2 = None
                l2g_t2 = None
            else:
                if True:
                    track_instances = self.test_track_instances
                time_delta = timestamp[0] - self.timestamp
                l2g_r1 = self.l2g_r_mat
                l2g_t1 = self.l2g_t
                l2g_r2 = l2g_r_mat
                l2g_t2 = l2g_t
        self.timestamp = timestamp[-1]
        self.l2g_r_mat = l2g_r_mat
        self.l2g_t = l2g_t

        # for bs 1;
        lidar2img = img_metas[0]['lidar2img']  # [T, num_cam]
        for i in range(num_frame):
            points_single = [p_[i] for p_ in points]
            img_single = torch.stack([img_[i] for img_ in img], dim=0)
            radar_single = torch.stack([radar_[i] for radar_ in radar], dim=0)

            img_metas_single = deepcopy(img_metas)
            img_metas_single[0]['lidar2img'] = lidar2img[i]

            track_instances = self._inference_single(points_single, img_single,
                                                     radar_single,
                                                     img_metas_single,
                                                     track_instances,
                                                     l2g_r1, l2g_t1, l2g_r2, l2g_t2,
                                                     time_delta,
                                                     gt_bboxes_3d=predictor_utils.tensors_tracking_to_detection(gt_bboxes_3d, i),
                                                     gt_labels_3d=predictor_utils.tensors_tracking_to_detection(gt_labels_3d, i))

            if True:
                track_instances = Instances.cat([track_instances, self._generate_empty_tracks()])

        # why again? select active has been performed in forward of qim.py
        active_instances = self.query_interact._select_active_tracks(
            dict(track_instances=track_instances))
        self.test_track_instances = track_instances

        results = self._active_instances2results(active_instances, img_metas)

        if mapping is not None:
            mapping = mapping[0]

        if self.do_pred and results[0] is not None and mapping['valid_pred'] \
                and (len(results[0]['track_ids']) > 0 and len(instance_idx_2_labels[0]) > 0):
            # only predict agent appeared at current index

            instance_idx_2_labels = instance_idx_2_labels[0]

            future_frame_num = 12

            index = mapping['index']
            valid_pred = mapping['valid_pred']

            if not hasattr(self, 'frame_index_2_mapping'):
                self.frame_index_2_mapping = {}
            self.frame_index_2_mapping[index] = mapping
            if index - 20 in self.frame_index_2_mapping:
                self.frame_index_2_mapping.pop(index - 20)

            if not hasattr(self, 'track_idx_2_boxes'):
                self.track_idx_2_boxes = defaultdict(dict)

            if True:
                # order of active_instances is different from results
                boxes_3d = predictor_utils.to_numpy(results[0]['boxes_3d'].tensor)
                output_embedding = results[0]['output_embedding']
                track_scores = predictor_utils.to_numpy(results[0]['track_scores'])
                track_ids = predictor_utils.to_numpy(results[0]['track_ids'])
                track_labels = predictor_utils.to_numpy(results[0]['labels_3d'])

            # 'boxes_3d' is in lidar, 'self.track_idx_2_boxes' is in global
            predictor_utils.update_track_idx_2_boxes(self.track_idx_2_boxes, track_ids, boxes_3d, mapping, index)

            # Input 'self.track_idx_2_boxes' is in global, output 'tracked_boxes_list' is in ego
            tracked_scores, tracked_trajs, tracked_boxes_list, tracked_boxes_is_valid_list, categories = \
                predictor_utils.extract_from_track_idx_2_boxes(self.track_idx_2_boxes, track_scores, track_ids, track_labels, mapping, index)

            gt_past_trajs, gt_past_trajs_is_valid, gt_future_trajs, gt_future_trajs_is_valid, gt_categories = \
                predictor_utils.get_gt_past_future_trajs(instance_idx_2_labels)

            labels, labels_is_valid = predictor_utils.get_labels_for_tracked_trajs(tracked_trajs, tracked_boxes_is_valid_list,
                                                                                   gt_past_trajs, gt_past_trajs_is_valid,
                                                                                   gt_future_trajs, gt_future_trajs_is_valid,
                                                                                   future_frame_num)

            if not valid_pred or len(tracked_boxes_list) == 0:
                outputs = None
            else:
                if output_embedding is not None:
                    if self.add_branch:
                        query = self.add_branch_update_query(output_embedding, boxes_3d[:, :3], device)
                        output_embedding = output_embedding + query

                    output_embedding = self.output_embedding_forward(output_embedding)

                if True:
                    loss, outputs, _ = self.predictor(agents=output_embedding.unsqueeze(0),
                                                      device=device,
                                                      labels=[labels],
                                                      labels_is_valid=[labels_is_valid],
                                                      agents_indices=None,
                                                      mapping=[mapping],
                                                      **kwargs)

            pred_dict = None
            if outputs is not None:
                pred_outputs = outputs['pred_outputs'][0]
                pred_probs = outputs['pred_probs'][0]
                pred_outputs_single_traj = []

                if self.relative_pred:
                    normalizers = [predictor_utils.get_normalizer(tracked_boxes_is_valid_list[j], tracked_boxes_list[j])
                                   for j in range(len(tracked_boxes_list))]
                    assert len(normalizers) == len(pred_outputs)
                    for j in range(len(normalizers)):
                        np.set_printoptions(precision=3, suppress=True)
                        argmax = np.argmax(pred_probs[j])
                        pred_outputs_single_traj.append(pred_outputs[j, argmax])
                        pred_outputs[j] = normalizers[j](pred_outputs[j], reverse=True)

                self.add_pred_results(results[0], pred_outputs, pred_probs)

                pred_dict = dict(
                    instance_idx_2_labels=instance_idx_2_labels,
                    pred_outputs=pred_outputs,
                    pred_probs=pred_probs,
                    lanes=mapping['lanes'],
                    tracked_trajs=np.array(tracked_trajs),
                    tracked_trajs_is_valid=np.array(tracked_boxes_is_valid_list),
                )

        if results[0] is not None and 'output_embedding' in results[0]:
            results[0].pop('output_embedding')

        return results

    def add_pred_results(self, result_dict, pred_outputs, pred_probs):
        result_dict['pred_outputs_in_ego'] = pred_outputs
        result_dict['pred_probs_in_ego'] = pred_probs

    def _active_instances2results(self, active_instances, img_metas, do_train=False):
        '''
        Outs:
        active_instances. keys:
        - 'pred_logits':
        - 'pred_boxes': normalized bboxes
        - 'scores'
        - 'obj_idxes'
        out_dict. keys:

            - boxes_3d (torch.Tensor): 3D boxes.
            - scores (torch.Tensor): Prediction scores.
            - labels_3d (torch.Tensor): Box labels.
            - attrs_3d (torch.Tensor, optional): Box attributes.
            - track_ids
            - tracking_score
        '''
        if do_train:
            assert False
            active_idxes = (active_instances.obj_idxes >= 0)
        else:
            # filter out sleep querys
            active_idxes = (active_instances.scores >= self.track_base.filter_score_thresh)
        active_instances = active_instances[active_idxes]
        if active_instances.pred_logits.numel() == 0:
            return [None]
        bbox_dict = dict(
            cls_scores=active_instances.pred_logits,
            bbox_preds=active_instances.pred_boxes,
            track_scores=active_instances.scores,
            obj_idxes=active_instances.obj_idxes,
            output_embedding=active_instances.output_embedding,
        )
        # plugin/track/bbox_coder.py
        bboxes_dict = self.bbox_coder.decode(bbox_dict)[0]

        bboxes = bboxes_dict['bboxes']
        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
        # (Pdb) img_metas[0]['box_type_3d'][0]
        # <class 'mmdet3d.core.bbox.structures.lidar_box3d.LiDARInstance3DBoxes'>
        bboxes = img_metas[0]['box_type_3d'][0](bboxes, 9)
        labels = bboxes_dict['labels']
        scores = bboxes_dict['scores']

        track_scores = bboxes_dict['track_scores']
        obj_idxes = bboxes_dict['obj_idxes']
        result_dict = dict(
            boxes_3d=bboxes.to('cpu'),
            scores_3d=scores.cpu(),
            labels_3d=labels.cpu(),
            track_scores=track_scores.cpu(),
            track_ids=obj_idxes.cpu(),
            output_embedding=bboxes_dict['output_embedding']
        )

        return [result_dict]

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the logger.
                - ``num_samples`` indicates the batch size (when the model is DDP, it means the batch size on each GPU), which is used for averaging the logs.
        """
        losses, others_dict = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        outputs.update(others_dict)

        return outputs

    def output_embedding_forward(self, output_embedding):
        if self.agents_layer_0:
            output_embedding = self.agents_layer_mlp_0(output_embedding)
        return output_embedding

    def update_history_img_list(self, img_metas, img, img_feats):
        if not hasattr(self, 'history_img_feats') or self.history_img_feats is None:
            self.history_img_feats = []
            self.history_img_metas = []
            self.history_img_list = []

        if True:
            self.history_img_feats.append(img_feats)
            self.history_img_metas.append(img_metas)
            self.history_img_list.append(img)

        if len(self.history_img_feats) > 3:
            self.history_img_feats = self.history_img_feats[1:]
            self.history_img_metas = self.history_img_metas[1:]
            self.history_img_list = self.history_img_list[1:]

    def add_branch_update_query(self, output_embedding, reference_points, device):
        reference_points = predictor_utils.reference_points_lidar_to_relative(reference_points, self.pc_range)
        reference_points = torch.tensor(reference_points, dtype=torch.float, device=device)

        query = self.add_branch_mlp(output_embedding)
        query = self.add_branch_attention(query=query.unsqueeze(1),
                                          reference_points=reference_points.unsqueeze(0),
                                          value=self.history_img_feats[-1],
                                          img_metas=self.history_img_metas[-1])
        assert query.shape == (len(output_embedding), 1, 256)
        query = query.squeeze(1)
        return query
