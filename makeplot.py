#!/usr/bin/env python
import math
import matplotlib.pyplot as plt
import numpy as np

nDimension, nPattern, nRun, learning_rate, energy_scale_lLTP, energy_scale_maintenance, synapse_threshold, arr_decay_eLTP = np.loadtxt(
    "Text/variables.txt")
decay_rates = np.loadtxt("Text/decay_rates.txt")
learning_rates = np.loadtxt("Text/learning_rates.txt")
nRun = int(synapse_threshold)

arr_color = ["black", "red", "lawngreen", "cyan", "purple"]
marker_types = ['.', ',', 'o', 'v', 'd']

xvalue = np.linspace(1, nRun, nRun)

###### Plot energy #######
marker_legends = []
for i in range(len(decay_rates)):
    arr_energy = np.loadtxt("Text/energy_" + str(decay_rates[i]) + ".txt")
    arr_energy_eLTP = np.loadtxt("Text/energy_eLTP_" + str(decay_rates[i]) + ".txt")
    arr_energy_lLTP = np.loadtxt("Text/energy_lLTP_" + str(decay_rates[i]) + ".txt")
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
plt.savefig("Plot/energy_d.png")
plt.close()

###### Plot epochs (Learning time) #######
for i in range(len(decay_rates)):
    arr_epoch = np.loadtxt("Text/epoch_" + str(decay_rates[i]) + ".txt")
    plt.plot(xvalue, arr_epoch, color=arr_color[i], linewidth=1, label="Decay rate = " + str(decay_rates[i]))
plt.xlabel("Threshold", fontsize=16)
plt.ylabel("Epochs", fontsize=16)
plt.xlim(0, nRun + 0.5)
plt.legend(fontsize=10, loc=3)
plt.tight_layout()
plt.savefig("Plot/epochs_d.png")
plt.close()

###### Plot error #######
for i in range(len(decay_rates)):
    arr_error = np.loadtxt("Text/error_" + str(decay_rates[i]) + ".txt")
    plt.plot(xvalue, arr_error, color=arr_color[i], linewidth=1, label="Decay rate = " + str(decay_rates[i]))
plt.xlabel("Threshold", fontsize=16)
plt.ylabel("Error", fontsize=16)
plt.xlim(0, nRun + 0.5)
plt.legend(fontsize=10, loc=2)
plt.tight_layout()
plt.savefig("Plot/error_d.png")
plt.close()

###### Plot energy #######
marker_legends = []
for i in range(len(learning_rates)):
    arr_energy = np.loadtxt("Text/energy_" + str(learning_rates[i]) + ".txt")
    arr_energy_eLTP = np.loadtxt("Text/energy_eLTP_" + str(learning_rates[i]) + ".txt")
    arr_energy_lLTP = np.loadtxt("Text/energy_lLTP_" + str(learning_rates[i]) + ".txt")
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
plt.savefig("Plot/energy_l.png")
plt.close()

###### Plot epochs (Learning time) #######
for i in range(len(learning_rates)):
    arr_epoch = np.loadtxt("Text/epoch_" + str(learning_rates[i]) + ".txt")
    plt.plot(xvalue, arr_epoch, color=arr_color[i], linewidth=1, label="Learning rate = " + str(learning_rates[i]))
plt.xlabel("Threshold", fontsize=16)
plt.ylabel("Epochs", fontsize=16)
plt.xlim(0, nRun + 0.5)
plt.legend(fontsize=10, loc=2)
plt.tight_layout()
plt.savefig("Plot/epochs_l.png")
plt.close()

###### Plot error #######
for i in range(len(learning_rates)):
    arr_error = np.loadtxt("Text/error_" + str(learning_rates[i]) + ".txt")
    plt.plot(xvalue, arr_error, color=arr_color[i], linewidth=1, label="Learning rate = " + str(learning_rates[i]))
plt.xlabel("Threshold", fontsize=16)
plt.ylabel("Error", fontsize=16)
plt.xlim(0, nRun + 0.5)
plt.legend(fontsize=10, loc=2)
plt.tight_layout()
plt.savefig("Plot/error_l.png")
plt.close()
