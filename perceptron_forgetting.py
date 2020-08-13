import Code.perceptron_helper as helper
import Code.makeplot as makeplot
import Code.MyConstants as Constants


def main():
    dir_name = '/new_patt_10'
    # helper.perceptron_forgetting_wrapper(nDimension=1000, nPattern=1000, dir_name=dir_name, n_iter=30)

    output_path = Constants.PERM_DECAY_FORGETTING_PATH + dir_name
    plot_path = Constants.PERM_DECAY_FORGETTING_PLOT_PATH + dir_name
    makeplot.plot_forgetting(output_path, plot_path)


if __name__ == '__main__':
    main()