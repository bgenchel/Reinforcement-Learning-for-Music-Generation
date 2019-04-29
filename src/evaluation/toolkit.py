import argparse
import os
import glob
import os.path as op
from mgeval.mgeval import core, utils
from sklearn.model_selection import LeaveOneOut
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

PRETRAINED_DIR = "../model/eval/eval_pretrained"
FULLY_TRAINED_DIR = "../model/eval/eval_fully_trained"
TARGET_DIR = "../model/eval/eval_reference"

METRICS = {"total_used_pitch": {"args": (), "kwargs": {}, "shape": (1,)},
           "bar_used_pitch": {"args": (), "kwargs": {"track_num": 1, "num_bar": 12}, "shape": (12, 1)},
           "total_used_note": {"args": (), "kwargs": {"track_num": 1}, "shape": (1,)},
           "bar_used_note": {"args": (), "kwargs": {"track_num": 1, "num_bar": 12}, "shape": (12, 1)},
           "total_pitch_class_histogram": {"args": (), "kwargs": {}, "shape": (12, 12)},
           "bar_pitch_class_histogram": {"args": (), "kwargs": {"track_num": 1, "num_bar": 12}, "shape": (12, 12)},
           "pitch_class_transition_matrix": {"args": (), "kwargs": {}, "shape": (12, 12)},
           "pitch_range": {"args": (), "kwargs": {}, "shape": (1,)},
           "avg_pitch_shift": {"args": (), "kwargs": {"track_num": 1}, "shape": (1,)},
           "avg_IOI": {"args": (), "kwargs": {}, "shape": (1,)},
           "note_length_hist": {"args": (), "kwargs": {"track_num": 1}, "shape": (12,)},
           "note_length_transition_matrix": {"args": (), "kwargs": {"track_num": 1}, "shape": (12, 12)}}


class MGEval:
    """
    Wrapper around Richard Yang's MGEval
    """

    def __init__(self, pred_dir, target_dir):
        self.pred_set = glob.glob(op.join(pred_dir, "*.mid"))
        self.target_set = glob.glob(op.join(target_dir, "*.mid"))

        pred_samples = len(self.pred_set)
        target_samples = len(self.target_set)

        if pred_samples > target_samples:
            self.num_samples = target_samples
        else:
            self.num_samples = pred_samples

        self.num_samples = 100

        self.metrics = core.metrics()

    def get_metric(self, metric_name, pred_metric_shape, target_metric_shape, *args, **kwargs):
        pred_metric = np.zeros((self.num_samples,) + pred_metric_shape)
        target_metric = np.zeros((self.num_samples,) + target_metric_shape)

        for sample in range(self.num_samples):
            pred_metric[sample] = getattr(self.metrics, metric_name)(core.extract_feature(self.pred_set[sample]), *args, **kwargs)
            target_metric[sample] = getattr(self.metrics, metric_name)(core.extract_feature(self.target_set[sample]), *args, **kwargs)

        return pred_metric, target_metric

    def inter_set_cross_validation(self, pred_metric, target_metric):
        loo = LeaveOneOut()
        loo.get_n_splits(np.arange(self.num_samples))

        inter_set_distance_cv = np.zeros((self.num_samples, 1, self.num_samples))

        for train_index, test_index in loo.split(np.arange(self.num_samples)):
            inter_set_distance_cv[test_index[0]][0] = utils.c_dist(pred_metric[test_index], target_metric)

        return inter_set_distance_cv

    def intra_set_cross_validation(self, pred_metric, target_metric):
        loo = LeaveOneOut()
        loo.get_n_splits(np.arange(self.num_samples))

        pred_intra_set_distance_cv = np.zeros((self.num_samples, 1, self.num_samples - 1))
        target_intra_set_distance_cv = np.zeros((self.num_samples, 1, self.num_samples - 1))

        for train_index, test_index in loo.split(np.arange(self.num_samples)):
            pred_intra_set_distance_cv[test_index[0]][0] = utils.c_dist(pred_metric[test_index],
                                                                        pred_metric[train_index])
            target_intra_set_distance_cv[test_index[0]][0] = utils.c_dist(target_metric[test_index],
                                                                          target_metric[train_index])

        return pred_intra_set_distance_cv, target_intra_set_distance_cv

    def visualize(self, metric_name, pred_intra, target_intra, inter):
        for measurement, label in zip([pred_intra, target_intra, inter], ["pred_intra", "target_intra", "inter"]):
            transposed = np.transpose(measurement, (1, 0, 2)).reshape(1, -1)
            sns.kdeplot(transposed[0], label=label)

        plt.title(metric_name)
        plt.xlabel('Euclidean distance')
        plt.show()

    def intra_inter_difference(self, metric_name, pred_intra, target_intra, inter):
        transposed = []

        for measurement, label in zip([pred_intra, target_intra, inter], ["pred_intra", "target_intra", "inter"]):
            transposed_meas = np.transpose(measurement, (1, 0, 2)).reshape(1, -1)
            transposed.append(transposed_meas)

        print(metric_name + ':')
        print('------------------------')
        print(' Predictions')
        print('  KL divergence:', utils.kl_dist(transposed[0][0], transposed[2][0]))
        print('  Overlap area:', utils.overlap_area(transposed[0][0], transposed[2][0]))
        print(' Targets')
        print('  KL divergence:', utils.kl_dist(transposed[1][0], transposed[2][0]))
        print('  Overlap area:', utils.overlap_area(transposed[1][0], transposed[2][0]))


def calculate_metric(mge, metric_name, pred_metric_shape, target_metric_shape, args, kwargs, statspath, figpath):
    # try:
    pred_metric, target_metric = mge.get_metric(metric_name, pred_metric_shape, 
                                                target_metric_shape, *args, **kwargs)
    inter = mge.inter_set_cross_validation(pred_metric, target_metric)
    pred_intra, target_intra = mge.intra_set_cross_validation(pred_metric, target_metric)
    mge.intra_inter_difference(metric_name, pred_intra, target_intra, inter, statspath)
    mge.visualize(metric_name, pred_intra, target_intra, inter, figpath)
    # except Exception as e:
        # print("Error occured while calculating {}: {}".format(metric_name, repr(e)))


def main(model, metric_name):
    if model == 'pretrained':
        mge = MGEval(PRETRAINED_DIR, TARGET_DIR)
    else:
        mge = MGEval(FULLY_TRAINED_DIR, TARGET_DIR)

    if metric_name != "all":
        statspath = op.join(os.getcwd(), 'mgeval_results', model, metric_name)
        figpath = op.join(os.getcwd(), 'mgeval_results', model, metric_name + '.png')
        if not op.exists(op.dirname(statspath)):
            os.makedirs(op.dirname(statspath))
        if not op.exists(op.dirname(figpath)):
            os.makedirs(op.dirname(figpath))
        calculate_metric(mge, metric_name, METRICS[metric_name]["shape"], METRICS[metric_name]["shape"],
                         METRICS[metric_name]["args"], METRICS[metric_name]["kwargs"], statspath, figpath)
    else:
        for k, v in METRICS.items():
            statspath = op.join(os.getcwd(), 'mgeval_results', model, k + '.txt')
            figpath = op.join(os.getcwd(), 'mgeval_results', model, k + '.png')
            if not op.exists(op.dirname(statspath)):
                os.makedirs(op.dirname(statspath))
            if not op.exists(op.dirname(figpath)):
                os.makedirs(op.dirname(figpath))
            calculate_metric(mge, k, v["shape"], v["shape"], v["args"], v["kwargs"], statspath, figpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default="fully_trained", type=str, 
                        choices=("pretrained", "fully_trained"), help="which model to evaluate.")
    parser.add_argument('-mt', '--metric', default="all", type=str,
                        choices=("total_used_pitch", "bar_used_pitch", "total_used_note",
                                 "bar_used_note", "total_pitch_class_histogram", "bar_pitch_class_histogram",
                                 "pitch_class_transition_matrix", "pitch_range", "avg_pitch_shift",
                                 "avg_IOI", "note_length_hist", "note_length_transition_matrix", "all"),
                        help="which model to evaluate.")
    args = parser.parse_args()
    main(args.model, args.metric)
