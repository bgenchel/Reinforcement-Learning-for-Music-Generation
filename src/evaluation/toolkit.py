import sys

sys.path.append("./mgeval")

import glob
import os.path as op
from mgeval import core, utils
from sklearn.model_selection import LeaveOneOut
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pdb


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


if __name__ == "__main__":
    mge = MGEval("../models/original_nottingham/eval_reference", "../models/original_nottingham/eval_fully_trained")
    # Expected shape of desired metric
    metric_shape = (12, 12)

    # Metric name
    metric_name = "note_length_transition_matrix"

    # Args and kwargs if needed for the desired metric
    args = ()
    kwargs = { "track_num": 1 }

    pred_metric, target_metric = mge.get_metric(metric_name, metric_shape, metric_shape, *args, **kwargs)
    inter = mge.inter_set_cross_validation(pred_metric, target_metric)
    pred_intra, target_intra = mge.intra_set_cross_validation(pred_metric, target_metric)
    mge.visualize(metric_name, pred_intra, target_intra, inter)
