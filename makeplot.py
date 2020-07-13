#!/usr/bin/env python
import math
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

nDimension, nPattern, nRun, learning_rate, energy_scale_lLTP, energy_scale_maintenance, synapse_threshold, arr_decay_eLTP = np.loadtxt(
    "../Text/variables.txt")
decay_rates = np.loadtxt("../Text/decay_rates.txt")
learning_rates = np.loadtxt("../Text/learning_rates.txt")
maintenance_costs = np.loadtxt("../Text/maintenance_costs.txt")
nRun = int(synapse_threshold)

arr_color = ["black", "red", "lawngreen", "cyan", "purple"]
marker_types = ['.', ',', 'o', 'v', 'd']

xvalue = np.linspace(1, nRun, nRun)

def plot_with_syn_threshold():
    ###### Plot energy #######
    marker_legends = []
    for i in range(len(decay_rates)):
        arr_energy = np.loadtxt("../Text/energy_" + str(decay_rates[i]) + ".txt")
        arr_energy_eLTP = np.loadtxt("../Text/energy_eLTP_" + str(decay_rates[i]) + ".txt")
        arr_energy_lLTP = np.loadtxt("../Text/energy_lLTP_" + str(decay_rates[i]) + ".txt")
        leg, = plt.plot(xvalue, arr_energy, color=arr_color[i], linewidth=1, label="Decay rate = " + str(decay_rates[i]))
        # plt.plot(xvalue, arr_energy_eLTP, color=arr_color[1], linewidth=1, marker=marker_types[i])
        # plt.plot(xvalue, arr_energy_lLTP, color=arr_color[2], linewidth=1, marker=marker_types[i])
        marker_legends.append(leg)
    # plt.plot(xvalue, arr_epoch, color=arr_color[3], linewidth=3, label="Epoch")
    plt.xlabel("Threshold", fontsize=16)
    plt.ylabel("Energy", fontsize=16)
    plt.yscale("log")
    plt.xlim(0, nRun + 0.5)
    plt.legend(handles=marker_legends, fontsize=10, loc=3)
    plt.tight_layout()
    plt.savefig("../Plot/energy_d.png")
    plt.close()

    ###### Plot epochs (Learning time) #######
    for i in range(len(decay_rates)):
        arr_epoch = np.loadtxt("../Text/epoch_" + str(decay_rates[i]) + ".txt")
        plt.plot(xvalue, arr_epoch, color=arr_color[i], linewidth=1, label="Decay rate = " + str(decay_rates[i]))
    plt.xlabel("Threshold", fontsize=16)
    plt.ylabel("Epochs", fontsize=16)
    plt.xlim(0, nRun + 0.5)
    plt.legend(fontsize=10, loc=3)
    plt.tight_layout()
    plt.savefig("../Plot/epochs_d.png")
    plt.close()

    ###### Plot error #######
    for i in range(len(decay_rates)):
        arr_error = np.loadtxt("../Text/error_" + str(decay_rates[i]) + ".txt")
        plt.plot(xvalue, arr_error, color=arr_color[i], linewidth=1, label="Decay rate = " + str(decay_rates[i]))
    plt.xlabel("Threshold", fontsize=16)
    plt.ylabel("Error", fontsize=16)
    plt.xlim(0, nRun + 0.5)
    plt.legend(fontsize=10, loc=2)
    plt.tight_layout()
    plt.savefig("../Plot/error_d.png")
    plt.close()

    ###### Plot energy #######
    marker_legends = []
    for i in range(len(learning_rates)):
        arr_energy = np.loadtxt("../Text/energy_" + str(learning_rates[i]) + ".txt")
        arr_energy_eLTP = np.loadtxt("../Text/energy_eLTP_" + str(learning_rates[i]) + ".txt")
        arr_energy_lLTP = np.loadtxt("../Text/energy_lLTP_" + str(learning_rates[i]) + ".txt")
        leg, = plt.plot(xvalue, arr_energy, color=arr_color[i], linewidth=1, label="Learning rate = " + str(learning_rates[i]))
        # plt.plot(xvalue, arr_energy_eLTP, color=arr_color[1], linewidth=1, marker=marker_types[i])
        # plt.plot(xvalue, arr_energy_lLTP, color=arr_color[2], linewidth=1, marker=marker_types[i])
        marker_legends.append(leg)
    # plt.plot(xvalue, arr_epoch, color=arr_color[3], linewidth=3, label="Epoch")
    plt.xlabel("Threshold", fontsize=16)
    plt.ylabel("Energy", fontsize=16)
    plt.yscale("log")
    plt.xlim(0, nRun + 0.5)
    plt.legend(handles=marker_legends, fontsize=10, loc=3)
    plt.tight_layout()
    plt.savefig("../Plot/energy_l.png")
    plt.close()

    ###### Plot epochs (Learning time) #######
    for i in range(len(learning_rates)):
        arr_epoch = np.loadtxt("../Text/epoch_" + str(learning_rates[i]) + ".txt")
        plt.plot(xvalue, arr_epoch, color=arr_color[i], linewidth=1, label="Learning rate = " + str(learning_rates[i]))
    plt.xlabel("Threshold", fontsize=16)
    plt.ylabel("Epochs", fontsize=16)
    plt.xlim(0, nRun + 0.5)
    plt.legend(fontsize=10, loc=2)
    plt.tight_layout()
    plt.savefig("../Plot/epochs_l.png")
    plt.close()

    ###### Plot error #######
    for i in range(len(learning_rates)):
        arr_error = np.loadtxt("../Text/error_" + str(learning_rates[i]) + ".txt")
        plt.plot(xvalue, arr_error, color=arr_color[i], linewidth=1, label="Learning rate = " + str(learning_rates[i]))
    plt.xlabel("Threshold", fontsize=16)
    plt.ylabel("Error", fontsize=16)
    plt.xlim(0, nRun + 0.5)
    plt.legend(fontsize=10, loc=2)
    plt.tight_layout()
    plt.savefig("../Plot/error_l.png")
    plt.close()


def plot_fig3():
    Path("../Plot/Fig3").mkdir(parents=True, exist_ok=True)

    # Plot energy
    for i in range(len(maintenance_costs)):
        arr_energy = np.loadtxt("../Text/Fig3/energy_" + str(maintenance_costs[i]) + ".txt")
        plt.plot(decay_rates, arr_energy, color=arr_color[i], linewidth=1, label="Maintenance cost = " + str(maintenance_costs[i]))
    plt.xlabel("Decay Rate", fontsize=16)
    plt.ylabel("Energy", fontsize=16)
    plt.legend(fontsize=10, loc=4)
    plt.tight_layout()
    plt.savefig("../Plot/Fig3/energy.png")
    plt.close()

    # Plot optimal threshold
    for i in range(len(maintenance_costs)):
        arr_opt_threshold = np.loadtxt("../Text/Fig3/opt_threshold_" + str(maintenance_costs[i]) + ".txt")
        plt.plot(decay_rates, arr_opt_threshold, color=arr_color[i], linewidth=1, label="Maintenance cost = " + str(maintenance_costs[i]))
    plt.xlabel("Decay Rate", fontsize=16)
    plt.ylabel("Opt threshold", fontsize=16)
    plt.legend(fontsize=10, loc=1)
    plt.tight_layout()
    plt.savefig("../Plot/Fig3/opt_threshold.png")
    plt.close()

    # Plot learning time
    for i in range(len(maintenance_costs)):
        arr_epoch = np.loadtxt("../Text/Fig3/epoch_" + str(maintenance_costs[i]) + ".txt")
        plt.plot(decay_rates, arr_epoch, color=arr_color[i], linewidth=1, label="Maintenance cost = " + str(maintenance_costs[i]))
    plt.xlabel("Decay Rate", fontsize=16)
    plt.ylabel("Learning time", fontsize=16)
    plt.legend(fontsize=10, loc=1)
    plt.tight_layout()
    plt.savefig("../Plot/Fig3/epoch.png")
    plt.close()


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
    plt.xticks(decay_rates_lLTP)
    plt.xlabel("Decay rate", fontsize=16)
    plt.ylabel("# Patterns", fontsize=16)
    plt.tight_layout()
    plt.savefig("../Plot/Perm_decay/patterns.png")
    plt.close()

    arr_energy = np.loadtxt("../Text/Perm_decay/energy.txt")
    plt.plot(decay_rates_lLTP, arr_energy, linewidth=1, marker='o', color=arr_color[4])
    plt.xticks(decay_rates_lLTP)
    plt.xlabel("Decay rate", fontsize=16)
    plt.ylabel("Energy", fontsize=16)
    plt.tight_layout()
    plt.savefig("../Plot/Perm_decay/energy.png")
    plt.close()

    arr_error = np.loadtxt("../Text/Perm_decay/error.txt")
    plt.plot(decay_rates_lLTP, arr_error, linewidth=1, marker='o', color=arr_color[4])
    plt.xticks(decay_rates_lLTP)
    plt.xlabel("Decay rate", fontsize=16)
    plt.ylabel("Error", fontsize=16)
    plt.tight_layout()
    plt.savefig("../Plot/Perm_decay/error.png")
    plt.close()

    arr_epoch = np.loadtxt("../Text/Perm_decay/epoch.txt")
    plt.plot(decay_rates_lLTP, arr_epoch, linewidth=1, marker='o', color=arr_color[4])
    plt.xticks(decay_rates_lLTP)
    plt.xlabel("Decay rate", fontsize=16)
    plt.ylabel("Epoch", fontsize=16)
    plt.tight_layout()
    plt.savefig("../Plot/Perm_decay/epoch.png")
    plt.close()


perm_decay_patterns()
plot_perm_decay_rates()
# plot_fig3()
