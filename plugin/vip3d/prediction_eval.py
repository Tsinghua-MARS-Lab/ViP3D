import argparse
import json
import os
import time
from typing import Tuple, List, Dict, Any

import numpy as np
from nuscenes import NuScenes
from nuscenes.eval.common.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.tracking.data_classes import TrackingMetrics, TrackingMetricDataList, TrackingConfig, TrackingBox


class PredictionMetrics:
    def __init__(self):
        self.history_frame_num = 3
        self.future_frame_num = 12

    def accumulate(self):
        pass

    def compute_metric(self, metric_name: str, class_name: str = 'all') -> float:
        if class_name == 'all':
            data = list(self.label_metrics[metric_name].values())
            if len(data) > 0:
                # Some metrics need to be summed, not averaged.
                # Nan entries are ignored.
                if metric_name in ['mt', 'ml', 'tp', 'fp', 'fn', 'ids', 'frag']:
                    return float(np.nansum(data))
                else:
                    return float(np.nanmean(data))
            else:
                return np.nan
        else:
            return float(self.label_metrics[metric_name][class_name])

    def serialize(self) -> Dict[str, Any]:
        metrics = dict()
        metrics

    @classmethod
    def deserialize(cls, content: dict) -> 'TrackingMetrics':
        """ Initialize from serialized dictionary. """
        cfg = TrackingConfig.deserialize(content['cfg'])
        tm = cls(cfg=cfg)
        tm.add_runtime(content['eval_time'])
        tm.label_metrics = content['label_metrics']

        return tm

    def __eq__(self, other):
        eq = True
        eq = eq and self.label_metrics == other.label_metrics
        eq = eq and self.eval_time == other.eval_time
        eq = eq and self.cfg == other.cfg

        return eq


class PredictionEval:
    def __init__(self,
                 cfg,
                 result_path: str,
                 eval_set: str,
                 output_dir: str,
                 nusc_version: str,
                 nusc_dataroot: str,
                 verbose: bool = True):
        """
        :param result_path: Path of the nuScenes JSON result file.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param output_dir: Folder to save plots and results to.
        :param nusc_version: The version of the NuScenes dataset.
        :param nusc_dataroot: Path of the nuScenes dataset on disk.
        :param verbose: Whether to print to stdout.
        """
        self.cfg = cfg
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose

        # Check result file exists.
        assert os.path.exists(result_path), 'Error: The result file does not exist!'

        # Make dirs.
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        # Initialize NuScenes object.
        # We do not store it in self to let garbage collection take care of it and save memory.
        nusc = NuScenes(version=nusc_version, verbose=verbose, dataroot=nusc_dataroot)

        # Load data.
        if verbose:
            print('Initializing prediction evaluation')
        pred_boxes, self.meta = load_prediction(self.result_path, 100, TrackingBox,
                                                verbose=verbose)

        gt_boxes = load_gt(nusc, self.eval_set, TrackingBox, verbose=verbose)

        assert set(pred_boxes.sample_tokens) == set(gt_boxes.sample_tokens), \
            "Samples in split don't match samples in predicted tracks."

        # Add center distances.
        pred_boxes = add_center_dist(nusc, pred_boxes)
        gt_boxes = add_center_dist(nusc, gt_boxes)

        # Filter boxes (distance, points per box, etc.).
        if verbose:
            print('Filtering tracks')
        self.pred_boxes = filter_eval_boxes(nusc, pred_boxes, self.cfg.class_range, verbose=verbose)
        if verbose:
            print('Filtering ground truth tracks')
        self.gt_boxes = filter_eval_boxes(nusc, gt_boxes, self.cfg.class_range, verbose=verbose)

        self.sample_tokens = self.gt_boxes.sample_tokens

    def evaluate(self):
        metrics = PredictionMetrics()

        return metrics

    def main(self) -> Dict[str, Any]:
        metrics: PredictionMetrics = self.evaluate()

        # Dump the metric data, meta and metrics to disk.
        if self.verbose:
            print('Saving metrics to: %s' % self.output_dir)
        metrics_summary = metrics.serialize()

        with open(os.path.join(self.output_dir, 'metrics_summary.json'), 'w') as f:
            json.dump(metrics_summary, f, indent=2)

        # Print metrics to stdout.
        if self.verbose:
            print(json.dumps(metrics_summary, indent=2))

        return metrics_summary
