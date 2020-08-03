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
    plt.plot(x, poly1d_fn(logx))
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
    coefficients = np.ones(shape=len(nDimensions))
    plt.style.use('seaborn-darkgrid')
    palette = plt.get_cmap('Dark2')

    for i in range(len(nDimensions)):
        print('N = ', nDimensions[i])
        output_path = Constants.PERM_DECAY_PATH + '/dim_' + str(nDimensions[i])
        decay_rates_lLTP = np.loadtxt(output_path + Constants.DECAY_RATES_FILE)
        arr_patterns = np.loadtxt(output_path + Constants.PATTERNS_FILE)

        # Fit a straight line to the plot
        x = np.array(decay_rates_lLTP)
        logx = np.log(x)
        # if nDimensions[i] == 250:
        #     arr_patterns = arr_patterns[8:]
        #     logx = logx[8:]
        #     x = x[8:]
        coeff = np.polyfit(logx, arr_patterns, 1)
        coefficients[i] = coeff[0]
        print('Coefficients:', coeff)
        poly1d_fn = np.poly1d(coeff)

        plt.figure(1)
        plt.plot(x, arr_patterns, color=palette(i), linestyle='-.', linewidth=2, label='N =' + str(nDimensions[i]))
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

    fig_patterns.savefig(Constants.BASE_PLOT_PATH + Constants.PATTERNS_PLOT)
    fig_energy.savefig(Constants.BASE_PLOT_PATH + Constants.ENERGY_PLOT)

    plt.figure(3)
    plt.plot(nDimensions, coefficients, color=palette(6))
    plt.xlabel('# Synapses')
    plt.ylabel('Slope')
    plt.tight_layout()
    plt.savefig(Constants.BASE_PLOT_PATH + Constants.PATTERNS_SLOPE_PLOT)


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


if __name__ == "__main__":
    main()
