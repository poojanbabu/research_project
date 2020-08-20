import Code.perceptron_helper as helper
import Code.makeplot as makeplot
import Code.MyConstants as Constants
import numpy as np


def main():
    decay_rates_lLTP = np.logspace(-7, -6, 11)
    dir_name = '/types_decay'
    helper.perceptron_forgetting_wrapper(nDimension=1000, nPattern=1000, dir_name=dir_name, new_patterns=100,
                                         n_iter=10, decay_rates_lLTP=decay_rates_lLTP)

    output_path = Constants.PERM_DECAY_FORGETTING_PATH + dir_name
    plot_path = Constants.PERM_DECAY_FORGETTING_PLOT_PATH + dir_name
    # makeplot.plot_forgetting(output_path, plot_path)
    makeplot.plot_forgetting_all_types(output_path, plot_path)


if __name__ == '__main__':
    main()
