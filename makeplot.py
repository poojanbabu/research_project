#!/usr/bin/env python
import math
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

nDimension, nRun, learning_rate, energy_scale_lLTP, energy_scale_maintenance, synapse_threshold, arr_decay_eLTP = np.loadtxt(
    "../Text/variables.txt")
decay_rates = np.loadtxt("../Text/decay_rates.txt")
learning_rates = np.loadtxt("../Text/learning_rates.txt")
maintenance_costs = np.loadtxt("../Text/maintenance_costs.txt")
nRun = int(synapse_threshold)

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

    decay_rates_lLTP = np.loadtxt("../Text/Perm_decay/decay_rates.txt")
    arr_patterns = np.loadtxt("../Text/Perm_decay/patterns.txt")
    plt.plot(decay_rates_lLTP, arr_patterns, linewidth=1, marker='o', color=arr_color[4])
    plt.xlabel("Decay rate", fontsize=16)
    plt.ylabel("# Patterns", fontsize=16)
    plt.tight_layout()
    plt.savefig("../Plot/Perm_decay/patterns.png")
    plt.close()

    arr_energy = np.loadtxt("../Text/Perm_decay/energy.txt")
    plt.plot(decay_rates_lLTP, arr_energy, linewidth=1, marker='o', color=arr_color[4])
    plt.xlabel("Decay rate", fontsize=16)
    plt.ylabel("Energy", fontsize=16)
    plt.tight_layout()
    plt.savefig("../Plot/Perm_decay/energy.png")
    plt.close()

    arr_error = np.loadtxt("../Text/Perm_decay/error.txt")
    plt.plot(decay_rates_lLTP, arr_error, linewidth=1, marker='o', color=arr_color[4])
    plt.xlabel("Decay rate", fontsize=16)
    plt.ylabel("Error", fontsize=16)
    plt.tight_layout()
    plt.savefig("../Plot/Perm_decay/error.png")
    plt.close()

    lineStyle = {"linestyle": "-", "linewidth": 2, "markeredgewidth": 2, "elinewidth": 2, "capsize": 3, "ecolor": "gray"}
    arr_epoch = np.loadtxt("../Text/Perm_decay/epoch.txt")
    arr_std_epoch = np.loadtxt("../Text/Perm_decay/std_epoch.txt")
    plt.errorbar(decay_rates_lLTP, arr_epoch, yerr=arr_std_epoch, **lineStyle, color=arr_color[4])
    plt.xlabel("Decay rate", fontsize=16)
    plt.ylabel("Epoch", fontsize=16)
    plt.tight_layout()
    plt.savefig("../Plot/Perm_decay/epoch.png")
    plt.close()


perm_decay_patterns()
plot_perm_decay_rates()
