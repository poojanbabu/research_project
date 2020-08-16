import Code.perceptron_helper as helper
import Code.makeplot as makeplot
import Code.MyConstants as Constants
import numpy as np


def main():
    decay_rates_lLTP = np.logspace(-7, -6, 10)
    for decay in decay_rates_lLTP:
        dir_name = '/new/' + str(decay)
        helper.perceptron_forgetting_wrapper(nDimension=1000, nPattern=1000, dir_name=dir_name, new_patterns=10,
                                             n_iter=30, decay_rate=decay)

        output_path = Constants.PERM_DECAY_FORGETTING_PATH + dir_name
        plot_path = Constants.PERM_DECAY_FORGETTING_PLOT_PATH + dir_name
        makeplot.plot_forgetting(output_path, plot_path)


if __name__ == '__main__':
    main()
