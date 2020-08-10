import Code.perceptron_helper as helper
import Code.MyConstants as Constants
import numpy as np


def main():
    ################### Max number of patterns vs decay rates ##############################
    nDimension = 250
    iPattern_init = 20
    step_size = 5
    decay_rates_lLTP = np.logspace(-2, -1, 10)
    dir_name = '/higher_decay'
    helper.perm_decay_wrapper(nDimension, iPattern_init, step_size, decay_rates_lLTP, dir_name)
    output_path_1 = Constants.PERM_DECAY_PATH + '/dim_' + str(nDimension) + '/higher_decay'
    output_path_2 = Constants.PERM_DECAY_PATH + '/dim_' + str(nDimension) + '/combined/old'
    res_output_path = Constants.PERM_DECAY_PATH + '/dim_' + str(nDimension) + '/combined'
    helper.combine_perceptron_decay_results(output_path_1, output_path_2, res_output_path, is_accuracy=False)


if __name__ == '__main__':
    main()
