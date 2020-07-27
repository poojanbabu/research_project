#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import OrderedDict

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


def perm_decay_patterns():
    Path("../Plot/Perm_decay").mkdir(parents=True, exist_ok=True)
    lineStyle = {"linestyle": "-", "linewidth": 2, "markeredgewidth": 2, "elinewidth": 1, "capsize": 3,
                 "ecolor": "gray"}

    # Plot the maximum number of patterns that a perceptron can learn for a decay value.
    decay_rates_lLTP = np.loadtxt("../Text/Perm_decay/decay_rates.txt")
    decay_rates_lLTP = np.concatenate((decay_rates_lLTP, np.loadtxt("../Text/Perm_decay_old/decay_rates.txt")))
    arr_patterns = np.loadtxt("../Text/Perm_decay/patterns.txt")
    arr_patterns = np.concatenate((arr_patterns, np.loadtxt("../Text/Perm_decay_old/patterns.txt")))
    dict_patterns = dict(zip(decay_rates_lLTP, arr_patterns))
    dict_patterns = OrderedDict(sorted(dict_patterns.items()))

    arr_std_patterns = np.loadtxt("../Text/Perm_decay/std_patterns.txt")
    arr_std_patterns = np.concatenate((arr_std_patterns, np.loadtxt("../Text/Perm_decay_old/std_patterns.txt")))
    dict_std_patterns = dict(zip(decay_rates_lLTP, arr_std_patterns))
    dict_std_patterns = OrderedDict(sorted(dict_std_patterns.items()))

    # Fit a straight line to the plot
    x = np.array([*dict_patterns.keys()])
    logx = np.log(x)
    y = np.array([*dict_patterns.values()])
    coeff = np.polyfit(logx, y, 1)
    poly1d_fn = np.poly1d(coeff)
    
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
    arr_energy = np.loadtxt("../Text/Perm_decay/energy.txt")
    arr_energy = np.concatenate((arr_energy, np.loadtxt("../Text/Perm_decay_old/energy.txt")))
    dict_energy = dict(zip(decay_rates_lLTP, arr_energy))
    dict_energy = OrderedDict(sorted(dict_energy.items()))

    arr_std_energy = np.loadtxt("../Text/Perm_decay/std_energy.txt")
    arr_std_energy = np.concatenate((arr_std_energy, np.loadtxt("../Text/Perm_decay_old/std_energy.txt")))
    dict_std_energy = dict(zip(decay_rates_lLTP, arr_std_energy))
    dict_std_energy = OrderedDict(sorted(dict_std_energy.items()))

    plt.errorbar([*dict_energy.keys()], [*dict_energy.values()], yerr=[*dict_std_energy.values()], **lineStyle,
                 color=arr_color[4])
    plt.xscale('log')
    plt.xlabel("Decay rate", fontsize=16)
    plt.ylabel("Energy", fontsize=16)
    plt.tight_layout()
    plt.savefig("../Plot/Perm_decay/energy.png")
    plt.close()

    # Plot the epochs for the max patterns trained
    arr_epoch = np.loadtxt("../Text/Perm_decay/epoch.txt")
    arr_epoch = np.concatenate((arr_epoch, np.loadtxt("../Text/Perm_decay_old/epoch.txt")))
    dict_epoch = dict(zip(decay_rates_lLTP, arr_epoch))
    dict_epoch = OrderedDict(sorted(dict_epoch.items()))

    arr_std_epoch = np.loadtxt("../Text/Perm_decay/std_epoch.txt")
    arr_std_epoch = np.concatenate((arr_std_epoch, np.loadtxt("../Text/Perm_decay_old/std_epoch.txt")))
    dict_std_epoch = dict(zip(decay_rates_lLTP, arr_std_epoch))
    dict_std_epoch = OrderedDict(sorted(dict_std_epoch.items()))

    plt.errorbar([*dict_epoch.keys()], [*dict_epoch.values()], yerr=[*dict_std_epoch.values()], **lineStyle, color=arr_color[4])
    plt.xscale('log')
    plt.xlabel("Decay rate", fontsize=16)
    plt.ylabel("Epoch", fontsize=16)
    plt.tight_layout()
    plt.savefig("../Plot/Perm_decay/epoch.png")
    plt.close()


perm_decay_patterns()
plot_perm_decay_rates()
