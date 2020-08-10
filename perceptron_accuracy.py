import numpy as np
import Code.MyConstants as Constants
import Code.perceptron_helper as helper
import Code.makeplot as makeplot


def main():
    nDimension = 1000
    nPattern = 1600
    decay_rates_lLTP = np.array([1e-4, 1e-3])
    dir_name = '/epoch_updates'
    helper.perceptron_accuracy_wrapper(nDimension, nPattern, decay_rates_lLTP, dir_name, epoch_updates=True)
    output_path = Constants.PERM_DECAY_ACCURACY_PATH + dir_name
    plot_path = Constants.PERM_DECAY_ACCURACY_PLOT_PATH + dir_name
    makeplot.plot_epoch_updates(output_path, plot_path)
    # output_path_1 = Constants.PERM_DECAY_ACCURACY_PATH + '/low_decay'
    # output_path_2 = Constants.PERM_DECAY_ACCURACY_PATH + '/combined'
    # res_output_path = Constants.PERM_DECAY_ACCURACY_PATH + '/combined'
    # combine_perceptron_decay_results(output_path_1, output_path_2, res_output_path, is_accuracy=True)


if __name__ == '__main__':
    main()
