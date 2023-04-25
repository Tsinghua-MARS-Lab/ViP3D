import argparse
import json
import os
from typing import List, Dict, Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm


class cfg:
    pred_traj_num = 6
    future_frame_num = 12
    max_dis_from_ego = 50.0
    matching_threshold = 2.0
    miss_rate_threshold = 2.0
    false_positive_penalty_coefficient = 0.5


class GTAgent:
    def __init__(self,
                 translation: np.ndarray = np.zeros(2),
                 future_traj: np.ndarray = np.zeros((cfg.future_frame_num, 2)),
                 future_traj_is_valid: np.ndarray = np.zeros(cfg.future_frame_num, dtype=np.int)
                 ):
        self.translation = translation.copy()
        self.future_traj = future_traj.copy()
        self.future_traj_is_valid = future_traj_is_valid.copy()


class PredAgent:
    def __init__(self,
                 sample_token: str = "",
                 translation: np.ndarray = np.zeros(2),
                 pred_future_trajs: np.ndarray = np.zeros((cfg.pred_traj_num, cfg.future_frame_num, 2)),
                 ):
        self.sample_token = sample_token
        self.translation = translation.copy()
        self.pred_future_trajs = pred_future_trajs.copy()

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized content. """

        if 'pred_outputs' in content:
            content['pred_future_trajs'] = content['pred_outputs']

        translation = np.array(content['translation'][:2])
        pred_future_trajs = np.array(content['pred_future_trajs'])

        return cls(
            translation=translation,
            pred_future_trajs=pred_future_trajs,
        )


class Metric:
    def __init__(self):
        self.values = []

    def accumulate(self, value):
        if value is not None:
            self.values.append(value)

    def get_mean(self):
        if len(self.values) > 0:
            return np.mean(self.values)
        else:
            return 0.0

    def get_sum(self):
        return np.sum(self.values)


class PredictionMetrics:
    def __init__(self):
        self.minADE = Metric()
        self.minFDE = Metric()
        self.MR = Metric()
        self.matched = Metric()
        self.unmatched = Metric()
        self.matched_and_prediction_hit = Metric()
        self.gt_agent_num = Metric()

    def serialize(self) -> Dict[str, Any]:
        unmatched = self.unmatched.get_sum()
        matched_and_prediction_hit = self.matched_and_prediction_hit.get_sum()
        gt_agent_num = self.gt_agent_num.get_sum()

        EPA = (matched_and_prediction_hit - unmatched * cfg.false_positive_penalty_coefficient) / gt_agent_num

        EPA = max(EPA, 0.0)

        return dict(
            minADE=self.minADE.get_mean(),
            minFDE=self.minFDE.get_mean(),
            MR=self.MR.get_mean(),
            EPA=EPA,
        )


def get_distances(point, points):
    assert point.ndim == 1 and points.ndim == 2
    return np.sqrt(np.square(points[:, 0] - point[0]) + np.square(points[:, 1] - point[1]))


def get_argmin_trajectory(future_traj, future_traj_is_valid, pred_future_trajs):
    if future_traj_is_valid.sum() == 0:
        return None, None, None

    delta: np.ndarray = pred_future_trajs - future_traj[np.newaxis, :]
    assert delta.shape == (cfg.pred_traj_num, cfg.future_frame_num, 2)

    delta = np.sqrt((delta * delta).sum(-1))
    assert delta.shape == (cfg.pred_traj_num, cfg.future_frame_num)

    if future_traj_is_valid[-1]:
        minFDE = delta[:, -1].min()
    else:
        minFDE = None

    delta = delta * future_traj_is_valid[np.newaxis, :]
    delta = delta.sum(-1) / future_traj_is_valid.sum()
    assert delta.shape == (cfg.pred_traj_num,)

    argmin = delta.argmin()
    minADE = delta.min()

    return argmin, minADE, minFDE


def get_gt_agents(prediction_infos, index):
    idx_2_gt_agent = {}

    for i in range(index, index + 1 + cfg.future_frame_num):
        if i >= len(prediction_infos):
            break

        info = prediction_infos[i]

        instance_inds = np.array(info['instance_inds'], dtype=np.int)
        gt_bboxes_3d = np.array(info['gt_bboxes_3d'])
        gt_labels_3d = np.array(info['gt_labels_3d'], dtype=np.int)

        assert len(instance_inds) == len(gt_bboxes_3d) == len(gt_labels_3d)

        for box_idx, instance_idx in enumerate(instance_inds):
            assert instance_idx != -1
            if i == index:
                assert instance_idx not in idx_2_gt_agent
                idx_2_gt_agent[instance_idx] = GTAgent()

            if instance_idx in idx_2_gt_agent:
                gt_prediction_box = idx_2_gt_agent[instance_idx]

                xy = gt_bboxes_3d[box_idx][:2]

                if i == index:
                    gt_prediction_box.translation[:] = xy
                else:
                    gt_prediction_box.future_traj[i - index - 1] = xy
                    gt_prediction_box.future_traj_is_valid[i - index - 1] = 1

    gt_agents = [value for key, value in idx_2_gt_agent.items()]
    return gt_agents


class PredictionEval:
    def __init__(self,
                 result_path: str = None,
                 output_dir: str = None,
                 prediction_infos_path: str = None):
        """
        Parameters
        ----------
        :param result_path: Path of the JSON result file.
        :param output_dir: Folder to save metrics.
        :param prediction_infos_path: Path of preprocessed gt boxes in JSON format
        """

        """
        Example of JSON result file:
        {
            sample_token_1: {
                {
                    translation: [900.0, 900.0],
                    pred_future_trajs: np.array of shape (cfg.pred_traj_num, cfg.future_frame_num, 2),
                },
                ...
                {
                    translation: [920.0, 920.0],
                    pred_future_trajs: np.array of shape (cfg.pred_traj_num, cfg.future_frame_num, 2),
                },
            },
            ...
            sample_token_n: {
                ...
            }
        }
        """

        if output_dir is None:
            # set to the directory of `result_path`
            output_dir = os.path.split(result_path)[0]

        self.output_dir = output_dir

        # Check result file exists.
        assert os.path.exists(result_path), 'Error: The result file does not exist!'

        # Make dirs.
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        with open(result_path) as f:
            data = json.load(f)
            if 'results' in data:
                data = data['results']

            self.sample_token_2_pred_agents = {}
            for sample_token, boxes in data.items():
                self.sample_token_2_pred_agents[sample_token] = [PredAgent.deserialize(each) for each in boxes]

        with open(prediction_infos_path, 'r') as f:
            prediction_infos = json.load(f)
            self.prediction_infos = [value for key, value in prediction_infos.items()]

        self.metrics = []
        self.minFDE_list = []

    def evaluate(self):
        metrics = PredictionMetrics()

        for index in tqdm(range(len(self.prediction_infos))):
            info = self.prediction_infos[index]
            sample_token = info['sample_token']
            if index > 0:
                assert self.prediction_infos[index]['timestamp'] > self.prediction_infos[index - 1]['timestamp']

            if sample_token not in self.sample_token_2_pred_agents:
                break

            gt_agents: List[GTAgent] = get_gt_agents(self.prediction_infos, index)
            pred_agents: List[PredAgent] = self.sample_token_2_pred_agents[sample_token]

            if len(gt_agents) > 0:

                matched_of_gt_box = np.ones(len(gt_agents), dtype=np.int) * -1
                cost_matrix = np.zeros((len(pred_agents), len(gt_agents)))
                gt_translations = np.array([each.translation for each in gt_agents])

                for i in range(len(pred_agents)):
                    cost_matrix[i] = get_distances(pred_agents[i].translation, gt_translations)
                    cost_matrix[i][np.nonzero(cost_matrix[i] > cfg.matching_threshold)] = 10000.0

                r_list, c_list = linear_sum_assignment(cost_matrix)

                for i in range(len(r_list)):
                    if cost_matrix[r_list[i], c_list[i]] <= cfg.matching_threshold:
                        matched_of_gt_box[c_list[i]] = r_list[i]

                matched = 0
                matched_and_prediction_hit = 0
                gt_not_valid = 0

                for i in range(len(gt_agents)):
                    box_idx = matched_of_gt_box[i]
                    gt_agent = gt_agents[i]

                    if box_idx == -1:
                        minADE = None
                        minFDE = None
                        MR = None
                    else:
                        matched += 1

                        pred_agent = pred_agents[box_idx]
                        argmin, minADE, minFDE = get_argmin_trajectory(gt_agent.future_traj, gt_agent.future_traj_is_valid, pred_agent.pred_future_trajs)

                        if minADE is not None and minADE > 100.0:
                            assert False, f'Error {minADE} is too large!'

                        if gt_agent.future_traj_is_valid[-1]:
                            assert minFDE is not None
                            MR = minFDE > cfg.miss_rate_threshold
                            if not MR:
                                matched_and_prediction_hit += 1
                        else:
                            MR = None

                    metrics.minADE.accumulate(minADE)
                    metrics.minFDE.accumulate(minFDE)
                    metrics.MR.accumulate(MR)

                    if not gt_agent.future_traj_is_valid[-1]:
                        gt_not_valid += 1

                metrics.matched.accumulate(matched)
                metrics.unmatched.accumulate(len(pred_agents) - matched)
                metrics.matched_and_prediction_hit.accumulate(matched_and_prediction_hit)
                metrics.gt_agent_num.accumulate(len(gt_agents) - gt_not_valid)

        return metrics

    def main(self) -> Dict[str, Any]:
        metrics: PredictionMetrics = self.evaluate()

        print(f'Saving metrics to: {self.output_dir}/prediction_metrics.json')
        metrics_summary = metrics.serialize()

        with open(os.path.join(self.output_dir, 'prediction_metrics.json'), 'w') as f:
            json.dump(metrics_summary, f, indent=4)

        print(json.dumps(metrics_summary, indent=4))

        return metrics_summary


def main():
    parser = argparse.ArgumentParser(description='Prediction evaluation')
    parser.add_argument('--result_path', help='path of prediction results in JSON format')
    parser.add_argument('--prediction_infos_path',
                        default='./nuscenes_prediction_infos_val.json',
                        help='path of preprocessed gt boxes in JSON format')
    args = parser.parse_args()

    nusc_eval = PredictionEval(result_path=args.result_path,
                               prediction_infos_path=args.prediction_infos_path)

    nusc_eval.main()


if __name__ == '__main__':
    main()
