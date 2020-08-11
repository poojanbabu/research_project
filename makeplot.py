#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import OrderedDict
import Code.MyConstants as Constants

arr_color = ["black", "red", "lawngreen", "cyan", "purple"]
marker_types = ['.', ',', 'o', 'v', 'd']


def plot_perm_decay_rates():
    Path("../Plot/Perm").mkdir(parents=True, exist_ok=True)

    # Plot energy
    nPatterns = np.loadtxt("../Text/patterns.txt")
    arr_energy = np.loadtxt("../Text/Perm/energy.txt")
    plt.plot(nPatterns, arr_energy, linewidth=1, color=arr_color[1], marker='o')
    plt.xticks(nPatterns)
    plt.yscale("log")
    plt.xlabel("# Patterns", fontsize=16)
    plt.ylabel("Energy", fontsize=16)
    plt.tight_layout()
    plt.savefig("../Plot/Perm/energy.png")
    plt.close()

    arr_error = np.loadtxt("../Text/Perm/error.txt")
    plt.plot(nPatterns, arr_error, linewidth=1, color=arr_color[1], marker='o')
    plt.xticks(nPatterns)
    plt.xlabel("# Patterns", fontsize=16)
    plt.ylabel("Error", fontsize=16)
    plt.tight_layout()
    plt.savefig("../Plot/Perm/error.png")
    plt.close()

    arr_epochs = np.loadtxt("../Text/Perm/epoch.txt")
    plt.plot(nPatterns, arr_epochs, linewidth=1, color=arr_color[1], marker='o')
    plt.xticks(nPatterns)
    plt.xlabel("# Patterns", fontsize=16)
    plt.ylabel("Epoch (Learning time)", fontsize=16)
    plt.tight_layout()
    plt.savefig("../Plot/Perm/epoch.png")
    plt.close()


def perm_decay_patterns_combine():
    Path("../Plot/Perm_decay").mkdir(parents=True, exist_ok=True)
    lineStyle = {"linestyle": "-", "linewidth": 2, "markeredgewidth": 2, "elinewidth": 1, "capsize": 3,
                 "ecolor": "gray"}

    # Plot the maximum number of patterns that a perceptron can learn for a decay value.
    decay_rates_lLTP = np.loadtxt("../Text/Perm_decay/dim_1000/decay_rates.txt")
    decay_rates_lLTP = np.concatenate((decay_rates_lLTP, np.loadtxt("../Text/Perm_decay_old/decay_rates.txt")))
    arr_patterns = np.loadtxt("../Text/Perm_decay/dim_1000/patterns.txt")
    arr_patterns = np.concatenate((arr_patterns, np.loadtxt("../Text/Perm_decay_old/patterns.txt")))
    dict_patterns = dict(zip(decay_rates_lLTP, arr_patterns))
    dict_patterns = OrderedDict(sorted(dict_patterns.items()))

    arr_std_patterns = np.loadtxt("../Text/Perm_decay/dim_1000/std_patterns.txt")
    arr_std_patterns = np.concatenate((arr_std_patterns, np.loadtxt("../Text/Perm_decay_old/std_patterns.txt")))
    dict_std_patterns = dict(zip(decay_rates_lLTP, arr_std_patterns))
    dict_std_patterns = OrderedDict(sorted(dict_std_patterns.items()))

    # Fit a straight line to the plot
    x = np.array([*dict_patterns.keys()])
    logx = np.log(x)
    y = np.array([*dict_patterns.values()])
    coeff = np.polyfit(logx, y, 1)
    print('Coefficients:', coeff)
    poly1d_fn = np.poly1d(coeff)

    np.savetxt('../Text/Perm_decay/dim_1000/decay_rates.txt', [*dict_patterns.keys()])
    np.savetxt('../Text/Perm_decay/dim_1000/patterns.txt', [*dict_patterns.values()])
    np.savetxt("../Text/Perm_decay/dim_1000/std_patterns.txt", [*dict_std_patterns.values()])
    plt.errorbar([*dict_patterns.keys()], [*dict_patterns.values()], yerr=[*dict_std_patterns.values()], **lineStyle,
                 color=arr_color[4])
    plt.plot(x, poly1d_fn(logx))
    plt.xscale('log')
    plt.xlabel("Decay rate", fontsize=16)
    plt.ylabel("# Patterns", fontsize=16)
    plt.tight_layout()
    plt.savefig("../Plot/Perm_decay/patterns.png")
    plt.close()

    # Plot the energy consumed for the max number of patterns trained
    arr_energy = np.loadtxt("../Text/Perm_decay/dim_1000/energy.txt")
    arr_energy = np.concatenate((arr_energy, np.loadtxt("../Text/Perm_decay_old/energy.txt")))
    dict_energy = dict(zip(decay_rates_lLTP, arr_energy))
    dict_energy = OrderedDict(sorted(dict_energy.items()))

    arr_std_energy = np.loadtxt("../Text/Perm_decay/dim_1000/std_energy.txt")
    arr_std_energy = np.concatenate((arr_std_energy, np.loadtxt("../Text/Perm_decay_old/std_energy.txt")))
    dict_std_energy = dict(zip(decay_rates_lLTP, arr_std_energy))
    dict_std_energy = OrderedDict(sorted(dict_std_energy.items()))

    np.savetxt('../Text/Perm_decay/dim_1000/energy.txt', [*dict_energy.values()])
    np.savetxt("../Text/Perm_decay/dim_1000/std_energy.txt", [*dict_std_energy.values()])

    plt.errorbar([*dict_energy.keys()], [*dict_energy.values()], yerr=[*dict_std_energy.values()], **lineStyle,
                 color=arr_color[4])
    plt.xscale('log')
    plt.xlabel("Decay rate", fontsize=16)
    plt.ylabel("Energy", fontsize=16)
    plt.tight_layout()
    plt.savefig("../Plot/Perm_decay/energy.png")
    plt.close()

    # Plot the epochs for the max patterns trained
    arr_epoch = np.loadtxt("../Text/Perm_decay/dim_1000/epoch.txt")
    arr_epoch = np.concatenate((arr_epoch, np.loadtxt("../Text/Perm_decay_old/epoch.txt")))
    dict_epoch = dict(zip(decay_rates_lLTP, arr_epoch))
    dict_epoch = OrderedDict(sorted(dict_epoch.items()))

    arr_std_epoch = np.loadtxt("../Text/Perm_decay/dim_1000/std_epoch.txt")
    arr_std_epoch = np.concatenate((arr_std_epoch, np.loadtxt("../Text/Perm_decay_old/std_epoch.txt")))
    dict_std_epoch = dict(zip(decay_rates_lLTP, arr_std_epoch))
    dict_std_epoch = OrderedDict(sorted(dict_std_epoch.items()))

    np.savetxt('../Text/Perm_decay/dim_1000/epoch.txt', [*dict_epoch.values()])
    np.savetxt("../Text/Perm_decay/dim_1000/std_epoch.txt", [*dict_std_epoch.values()])

    plt.errorbar([*dict_epoch.keys()], [*dict_epoch.values()], yerr=[*dict_std_epoch.values()], **lineStyle,
                 color=arr_color[4])
    plt.xscale('log')
    plt.xlabel("Decay rate", fontsize=16)
    plt.ylabel("Epoch", fontsize=16)
    plt.tight_layout()
    plt.savefig("../Plot/Perm_decay/epoch.png")
    plt.close()


def perm_decay_patterns(output_path, plot_path):
    Path(plot_path).mkdir(parents=True, exist_ok=True)
    lineStyle = {"linestyle": "-", "linewidth": 2, "markeredgewidth": 2, "elinewidth": 1, "capsize": 3,
                 "ecolor": "gray"}

    # Plot the maximum number of patterns that a perceptron can learn for a decay value.
    decay_rates_lLTP = np.loadtxt(output_path + Constants.DECAY_RATES_FILE)
    arr_patterns = np.loadtxt(output_path + Constants.PATTERNS_FILE)
    arr_std_patterns = np.loadtxt(output_path + Constants.STD_PATTERNS_FILE)

    # Fit a straight line to the plot
    x = np.array(decay_rates_lLTP)
    logx = np.log(x)
    coeff = np.polyfit(logx, arr_patterns, 1)
    print('Coefficients:', coeff)
    poly1d_fn = np.poly1d(coeff)

    plt.errorbar(x, arr_patterns, yerr=arr_std_patterns, **lineStyle, color=arr_color[4])
    # plt.plot(x, poly1d_fn(logx))
    plt.xscale('log')
    plt.xlabel("Decay rate", fontsize=16)
    plt.ylabel("# Patterns", fontsize=16)
    plt.tight_layout()
    plt.savefig(plot_path + Constants.PATTERNS_PLOT)
    plt.close()

    # Plot the energy consumed for the max number of patterns trained
    arr_energy = np.loadtxt(output_path + Constants.ENERGY_FILE)
    arr_std_energy = np.loadtxt(output_path + Constants.STD_ENERGY_FILE)

    plt.errorbar(decay_rates_lLTP, arr_energy, yerr=arr_std_energy, **lineStyle, color=arr_color[4])
    plt.xscale('log')
    plt.xlabel("Decay rate", fontsize=16)
    plt.ylabel("Energy", fontsize=16)
    plt.tight_layout()
    plt.savefig(plot_path + Constants.ENERGY_PLOT)
    plt.close()

    # Plot the epochs for the max patterns trained
    arr_epoch = np.loadtxt(output_path + Constants.EPOCH_FILE)
    arr_std_epoch = np.loadtxt(output_path + Constants.STD_EPOCH_FILE)

    plt.errorbar(decay_rates_lLTP, arr_epoch, yerr=arr_std_epoch, **lineStyle, color=arr_color[4])
    plt.xscale('log')
    plt.xlabel("Decay rate", fontsize=16)
    plt.ylabel("Epoch", fontsize=16)
    plt.tight_layout()
    plt.savefig(plot_path + Constants.EPOCH_PLOT)
    plt.close()


def perm_decay_patterns_nd(nDimensions):
    fig_patterns = plt.figure(1)
    fig_energy = plt.figure(2)
    fig_energy_per_pattern = plt.figure(3)
    fig_energy_per_synapse = plt.figure(4)
    coefficients = np.ones(shape=len(nDimensions))
    plt.style.use('seaborn-darkgrid')
    palette = plt.get_cmap('Dark2')

    for i in range(len(nDimensions)):
        print('N = ', nDimensions[i])
        if nDimensions[i] == 250:
            output_path = Constants.PERM_DECAY_PATH + '/dim_' + str(nDimensions[i]) + '/mid_decay'
        else:
            output_path = Constants.PERM_DECAY_PATH + '/dim_' + str(nDimensions[i])
        decay_rates_lLTP = np.loadtxt(output_path + Constants.DECAY_RATES_FILE)
        arr_patterns = np.loadtxt(output_path + Constants.PATTERNS_FILE)

        # Fit a straight line to the plot
        plt.figure(1)
        plt.plot(decay_rates_lLTP, arr_patterns, color=palette(i), linestyle='-.', linewidth=2,
                 label='N =' + str(nDimensions[i]))

        x = np.array(decay_rates_lLTP)
        logx = np.log(x)
        arr_patterns_fit = arr_patterns
        if nDimensions[i] == 250:
            arr_patterns_fit = arr_patterns[10:]
            logx = logx[10:]
            x = x[10:]
        coeff = np.polyfit(logx, arr_patterns_fit, 1)
        coefficients[i] = coeff[0]
        print('Coefficients:', coeff)
        poly1d_fn = np.poly1d(coeff)

        plt.plot(x, poly1d_fn(logx), color=palette(i))
        plt.xlabel('Decay rates')
        plt.ylabel('# Patterns')
        plt.xscale('log')
        plt.legend()
        plt.tight_layout()

        arr_energy = np.loadtxt(output_path + Constants.ENERGY_FILE)
        plt.figure(2)
        plt.plot(decay_rates_lLTP, arr_energy, color=palette(i), label='N = ' + str(nDimensions[i]))
        plt.xlabel('Decay rates')
        plt.ylabel('Energy')
        plt.xscale('log')
        plt.legend()
        plt.tight_layout()

        energy_per_pattern = np.divide(arr_energy, arr_patterns)
        plt.figure(3)
        plt.plot(decay_rates_lLTP, energy_per_pattern, color=palette(i), label='N = ' + str(nDimensions[i]))
        plt.xlabel('Decay rates')
        plt.ylabel('Energy per pattern')
        plt.xscale('log')
        plt.legend()
        plt.tight_layout()

        energy_per_synapse = np.divide(arr_energy, arr_patterns) / nDimensions[i]
        plt.figure(4)
        plt.plot(decay_rates_lLTP, energy_per_synapse, color=palette(i), label='N = ' + str(nDimensions[i]))
        plt.xlabel('Decay rates')
        plt.ylabel('Energy per pattern per synapse')
        plt.xscale('log')
        plt.legend()
        plt.tight_layout()

    fig_patterns.savefig(Constants.PERM_DECAY_PLOT_PATH + Constants.PATTERNS_PLOT)
    fig_energy.savefig(Constants.PERM_DECAY_PLOT_PATH + Constants.ENERGY_PLOT)
    fig_energy_per_pattern.savefig(Constants.PERM_DECAY_PLOT_PATH + Constants.ENERGY_PER_PATTERN_PLOT)
    fig_energy_per_synapse.savefig(Constants.PERM_DECAY_PLOT_PATH + Constants.ENERGY_PER_SYNAPSE_PLOT)

    coeff = np.polyfit(nDimensions, coefficients, 1)
    print('Coefficients:', coeff)
    poly1d_fn = np.poly1d(coeff)
    plt.figure(5)
    plt.scatter(nDimensions, coefficients, color='red', marker='x')
    plt.plot(nDimensions, poly1d_fn(nDimensions), color=palette(8))
    plt.xlabel('# Synapses')
    plt.ylabel('Slope')
    plt.tight_layout()
    plt.savefig(Constants.PERM_DECAY_PLOT_PATH + Constants.PATTERNS_SLOPE_PLOT)


def plot_perceptron_accuracy(output_path, plot_path, output_path_zero_decay=None):
    Path(plot_path).mkdir(parents=True, exist_ok=True)
    lineStyle = {"linestyle": "-", "linewidth": 2, "markeredgewidth": 2, "elinewidth": 1, "capsize": 3,
                 "ecolor": "gray"}

    decay_rates_lLTP = np.loadtxt(output_path + Constants.DECAY_RATES_FILE)
    arr_accuracy = np.loadtxt(output_path + Constants.ACCURACY_FILE)
    arr_error = np.loadtxt(output_path + Constants.ERROR_FILE)
    arr_energy = np.loadtxt(output_path + Constants.ENERGY_FILE)
    arr_epoch = np.loadtxt(output_path + Constants.EPOCH_FILE)

    arr_std_accuracy = np.loadtxt(output_path + Constants.STD_ACCURACY_FILE)
    arr_std_error = np.loadtxt(output_path + Constants.STD_ERROR_FILE)
    arr_std_energy = np.loadtxt(output_path + Constants.STD_ENERGY_FILE)
    arr_std_epoch = np.loadtxt(output_path + Constants.STD_EPOCH_FILE)

    plt.style.use('seaborn-darkgrid')
    palette = plt.get_cmap('Dark2')

    plt.errorbar(decay_rates_lLTP, arr_accuracy, yerr=arr_std_accuracy, **lineStyle, color=palette(0))
    plt.xlabel('Decay rates')
    plt.ylabel('Accuracy')
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(plot_path + Constants.ACCURACY_PLOT)
    plt.close()

    plt.errorbar(decay_rates_lLTP, arr_error, yerr=arr_std_error, **lineStyle, color=palette(1))
    plt.xlabel('Decay rates')
    plt.ylabel('Error')
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(plot_path + Constants.ERROR_PLOT)
    plt.close()

    if output_path_zero_decay is not None:
        energy_zero_decay = np.loadtxt(output_path_zero_decay + Constants.ENERGY_FILE)
        plt.axhline(y=energy_zero_decay, color='gray', lineStyle='--', linewidth=2, label='Decay rate: 0')
    plt.errorbar(decay_rates_lLTP, arr_energy, yerr=arr_std_energy, **lineStyle, color=palette(2))
    plt.xlabel('Decay rates')
    plt.ylabel('Energy')
    plt.xscale('log')
    plt.tight_layout()
    plt.legend()
    plt.savefig(plot_path + Constants.ENERGY_PLOT)
    plt.close()

    plt.errorbar(decay_rates_lLTP, arr_epoch, yerr=arr_std_epoch, **lineStyle, color=palette(3))
    plt.xlabel('Decay rates')
    plt.ylabel('Epoch')
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(plot_path + Constants.EPOCH_PLOT)
    plt.close()


def plot_epoch_updates(output_path, plot_path):
    # output_path = Constants.PERM_DECAY_PATH + '/accuracy_epoch_update'
    # plot_path = Constants.PERM_DECAY_PLOT_PATH + '/accuracy_epoch_update'
    Path(plot_path).mkdir(parents=True, exist_ok=True)
    plt.style.use('seaborn-darkgrid')
    palette = plt.get_cmap('Dark2')

    decay_rates_lLTP = np.loadtxt(output_path + Constants.DECAY_RATES_FILE)
    arr_epoch_updates = np.load(output_path + Constants.EPOCH_UPDATES_ALL, allow_pickle=True)
    arr_energy_updates = np.load(output_path + Constants.ENERGY_UPDATES_ALL, allow_pickle=True)

    for i in range(arr_epoch_updates.shape[0]):
        arr_epoch = arr_epoch_updates[i]
        arr_energy = arr_energy_updates[i]
        for j in range(arr_epoch.shape[0]):
            epochs = arr_epoch[j]
            plt.figure(1)
            plt.plot(range(len(epochs)), epochs, color=palette(i), label='Decay rate: ' + str(decay_rates_lLTP[i]))

            energy = arr_energy[j]
            plt.figure(2)
            plt.plot(range(len(energy)), energy, color=palette(i), label='Decay rate: ' + str(decay_rates_lLTP[i]))

    plt.figure(1)
    plt.xlabel('Epoch')
    plt.ylabel('# Updates')
    plt.tight_layout()
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.savefig(plot_path + Constants.EPOCH_UPDATES_PLOT)
    plt.close()

    plt.figure(2)
    plt.xlabel('Energy')
    plt.ylabel('# Updates')
    plt.tight_layout()
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.savefig(plot_path + Constants.ENERGY_UPDATES_PLOT)
    plt.close()

    arr_energy = np.loadtxt(output_path + Constants.ENERGY_FILE)
    plt.plot(decay_rates_lLTP, arr_energy, color=palette(4))
    plt.xlabel('Decay rates')
    plt.ylabel('Energy')
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(plot_path + Constants.ENERGY_PLOT)
    plt.close()

    arr_epoch = np.loadtxt(output_path + Constants.EPOCH_FILE)
    plt.plot(decay_rates_lLTP, arr_epoch, color=palette(4))
    plt.xlabel('Decay rates')
    plt.ylabel('Epoch')
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(plot_path + Constants.EPOCH_PLOT)
    plt.close()


def main():
    nDimensions = [1500, 1000, 500, 250]
    # for i in range(len(nDimensions)):
    #     nDimension = nDimensions[i]
    #     output_path = Constants.PERM_DECAY_PATH + '/dim_' + str(nDimension)
    #     plot_path = Constants.PERM_DECAY_PLOT_PATH + '/dim_' + str(nDimension)
    #     perm_decay_patterns(output_path, plot_path)

    # perm_decay_patterns_combine()
    # plot_perm_decay_rates()

    perm_decay_patterns_nd(nDimensions)

    # Accuracy and other measures for decay rates from 1e-6 to 1e-2
    # output_path_zero_decay = Constants.PERM_DECAY_ACCURACY_PATH + '/zero_decay'
    # output_path = Constants.PERM_DECAY_ACCURACY_PATH + '/combined'
    # plot_path = Constants.PERM_DECAY_ACCURACY_PLOT_PATH + '/combined'
    # plot_perceptron_accuracy(output_path, plot_path, output_path_zero_decay)

    # #patterns vs decay rates plot
    # output_path = Constants.PERM_DECAY_PATH + '/dim_' + str(250) + '/combined'
    # plot_path = Constants.PERM_DECAY_PLOT_PATH + '/dim_' + str(250) + '/combined'
    # perm_decay_patterns(output_path, plot_path)


if __name__ == "__main__":
    main()
