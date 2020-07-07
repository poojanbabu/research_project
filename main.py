#!/usr/bin/env python
import numpy as np
import MyFunction as MyF
from pathlib import Path

nDimension = 1000
nPattern = 1000
nRun = 10  # one file contains events with one hyperplane solution
learning_rate = 1.0
size_buffer = 1000
energy_scale_lLTP = 1.0
energy_scale_maintenance = energy_scale_lLTP * np.array([0.001])
synapse_threshold = 2
decay_eLTP = 0.001

synapse_threshold_max = 40

#####################################
#        Pattern generation         #
#####################################
pattern = np.array([None] * nRun)
pattern_answer = np.array([None] * nRun)
for iRun in range(nRun):
    pattern[iRun], pattern_answer[iRun] = MyF.MakePattern(nPattern, nDimension, is_pattern_integer=True,
                                                          is_plus_minus_one=True)


#####################################
#          Perceptron part          #
#####################################
def synapse_thr_with_decay_rates():
    decay_rates = np.arange(0, 0.005, 0.001)

    weight_initial = np.zeros(nDimension + 1)  # +1 is for bias
    Per = MyF.Perceptron(pattern[0], pattern_answer[0], weight_initial, \
                         size_buffer, learning_rate, energy_scale_lLTP, \
                         energy_detail=True)
    Per.energy_scale_maintenance = energy_scale_maintenance

    arr_energy = np.nan * np.ones(synapse_threshold_max)  # total energy
    arr_energy_eLTP = np.nan * np.ones(synapse_threshold_max)  # transient energy
    arr_energy_lLTP = np.nan * np.ones(synapse_threshold_max)  # permanent energy
    arr_error = np.nan * np.ones(synapse_threshold_max)  # number of errors
    arr_epoch = np.nan * np.ones(synapse_threshold_max)  # training time (epoch)

    for decay_rate in decay_rates:
        Per.decay_eLTP = decay_rate
        for i_synapse_threshold in range(synapse_threshold_max):
            energy = np.nan * np.ones(nRun)
            energy_eLTP = np.nan * np.ones(nRun)
            energy_lLTP = np.nan * np.ones(nRun)
            error = np.nan * np.ones(nRun)
            epoch = np.nan * np.ones(nRun)
            for iRun in range(nRun):
                print("Run: {}".format(iRun+1))
                Per.pattern = pattern[iRun]
                Per.pattern_answer = pattern_answer[iRun]
                energy[iRun], energy_eLTP[iRun], energy_lLTP[iRun], error[iRun], epoch[iRun] = Per.AlgoSynapseAll(i_synapse_threshold)

            arr_energy[i_synapse_threshold] = np.mean(energy)
            arr_energy_eLTP[i_synapse_threshold] = np.mean(energy_eLTP)
            arr_energy_lLTP[i_synapse_threshold] = np.mean(energy_lLTP)
            arr_error[i_synapse_threshold] = np.mean(error)
            arr_epoch[i_synapse_threshold] = np.mean(epoch)
            print("Threshold:", i_synapse_threshold, " energy:", arr_energy[i_synapse_threshold], " energy_eLTP:", arr_energy_eLTP[i_synapse_threshold], " energy_lLTP:",
                  arr_energy_lLTP[i_synapse_threshold], " error:", arr_error[i_synapse_threshold], " epoch:", arr_epoch[i_synapse_threshold])

        np.savetxt("../Text/energy_" + str(decay_rate) + ".txt", arr_energy)
        np.savetxt("../Text/energy_eLTP_" + str(decay_rate) + ".txt", arr_energy_eLTP)
        np.savetxt("../Text/energy_lLTP_" + str(decay_rate) + ".txt", arr_energy_lLTP)
        np.savetxt("../Text/error_" + str(decay_rate) + ".txt", arr_error)
        np.savetxt("../Text/epoch_" + str(decay_rate) + ".txt", arr_epoch)

    np.savetxt("../Text/decay_rates.txt", decay_rates)


def synapse_thr_with_learning_rates():
    learning_rates = np.array([0.1, 0.5, 1.0, 1.5, 2.0])

    weight_initial = np.zeros(nDimension + 1)  # +1 is for bias
    Per = MyF.Perceptron(pattern[0], pattern_answer[0], weight_initial, \
                         size_buffer, learning_rate, energy_scale_lLTP, \
                         energy_detail=True)
    Per.energy_scale_maintenance = energy_scale_maintenance
    Per.decay_eLTP = decay_eLTP

    arr_energy = np.nan * np.ones(synapse_threshold_max)  # total energy
    arr_energy_eLTP = np.nan * np.ones(synapse_threshold_max)  # transient energy
    arr_energy_lLTP = np.nan * np.ones(synapse_threshold_max)  # permanent energy
    arr_error = np.nan * np.ones(synapse_threshold_max)  # number of errors
    arr_epoch = np.nan * np.ones(synapse_threshold_max)  # training time (epoch)

    for i_learning_rate in learning_rates:
        Per.learning_rate = i_learning_rate
        for i_synapse_threshold in range(synapse_threshold_max):
            energy = np.nan * np.ones(nRun)
            energy_eLTP = np.nan * np.ones(nRun)
            energy_lLTP = np.nan * np.ones(nRun)
            error = np.nan * np.ones(nRun)
            epoch = np.nan * np.ones(nRun)
            for iRun in range(nRun):
                print("Pattern: {}".format(iRun))
                Per.pattern = pattern[iRun]
                Per.pattern_answer = pattern_answer[iRun]
                energy[iRun], energy_eLTP[iRun], energy_lLTP[iRun], error[iRun], epoch[iRun] = Per.AlgoSynapseAll(
                    i_synapse_threshold)

            arr_energy[i_synapse_threshold] = np.mean(energy)
            arr_energy_eLTP[i_synapse_threshold] = np.mean(energy_eLTP)
            arr_energy_lLTP[i_synapse_threshold] = np.mean(energy_lLTP)
            arr_error[i_synapse_threshold] = np.mean(error)
            arr_epoch[i_synapse_threshold] = np.mean(epoch)
            print("Threshold:", i_synapse_threshold, " energy:", arr_energy[i_synapse_threshold], " energy_eLTP:",
                  arr_energy_eLTP[i_synapse_threshold], " energy_lLTP:",
                  arr_energy_lLTP[i_synapse_threshold], " error:", arr_error[i_synapse_threshold], " epoch:",
                  arr_epoch[i_synapse_threshold])

        np.savetxt("../Text/energy_" + str(i_learning_rate) + ".txt", arr_energy)
        np.savetxt("../Text/energy_eLTP_" + str(i_learning_rate) + ".txt", arr_energy_eLTP)
        np.savetxt("../Text/energy_lLTP_" + str(i_learning_rate) + ".txt", arr_energy_lLTP)
        np.savetxt("../Text/error_" + str(i_learning_rate) + ".txt", arr_error)
        np.savetxt("../Text/epoch_" + str(i_learning_rate) + ".txt", arr_epoch)

    np.savetxt("../Text/learning_rates.txt", learning_rates)


def transient_decay_rates():
    decay_rates = np.arange(0, 0.005, 0.001)
    maintenance_costs = [0.001, 0.01]

    weight_initial = np.zeros(nDimension + 1)  # +1 is for bias
    Per = MyF.Perceptron(pattern[0], pattern_answer[0], weight_initial, \
                         size_buffer, learning_rate, energy_scale_lLTP, \
                         energy_detail=True)


    arr_energy = np.nan * np.ones(len(decay_rates))  # total energy
    arr_energy_eLTP = np.nan * np.ones(len(decay_rates))  # transient energy
    arr_energy_lLTP = np.nan * np.ones(len(decay_rates))  # permanent energy
    arr_error = np.nan * np.ones(len(decay_rates))  # number of errors
    arr_epoch = np.nan * np.ones(len(decay_rates))  # training time (epoch)
    arr_opt_threshold = np.nan * np.ones(len(decay_rates))  # optimal threshold

    Path("../Text/Fig3").mkdir(parents=True, exist_ok=True)
    for i_maintenance_cost in maintenance_costs:
        print("Maintenance cost: {}".format(i_maintenance_cost))
        Per.energy_scale_maintenance = i_maintenance_cost
        for idx in range(len(decay_rates)):
            Per.decay_eLTP = decay_rates[idx]
            arr_energy_syn = []
            arr_energy_eLTP_syn = []
            arr_energy_lLTP_syn = []
            arr_error_syn = []
            arr_epoch_syn = []
            print('Decay rate: {}'.format(decay_rates[idx]))
            for i_synapse_threshold in range(synapse_threshold_max):
                energy = np.nan * np.ones(nRun)
                energy_eLTP = np.nan * np.ones(nRun)
                energy_lLTP = np.nan * np.ones(nRun)
                error = np.nan * np.ones(nRun)
                epoch = np.nan * np.ones(nRun)
                for iRun in range(nRun):
                    Per.pattern = pattern[iRun]
                    Per.pattern_answer = pattern_answer[iRun]
                    energy[iRun], energy_eLTP[iRun], energy_lLTP[iRun], error[iRun], epoch[iRun] = Per.AlgoSynapseAll(i_synapse_threshold)

                if np.mean(error) > 0.0:
                    break

                arr_energy_syn.append(np.mean(energy))
                arr_energy_eLTP_syn.append(np.mean(energy_eLTP))
                arr_energy_lLTP_syn.append(np.mean(energy_lLTP))
                arr_error_syn.append(np.mean(error))
                arr_epoch_syn.append(np.mean(epoch))
                print("Threshold:", i_synapse_threshold, " energy:", arr_energy_syn[i_synapse_threshold], " energy_eLTP:", arr_energy_eLTP_syn[i_synapse_threshold], " energy_lLTP:",
                      arr_energy_lLTP_syn[i_synapse_threshold], " error:", arr_error_syn[i_synapse_threshold], " epoch:", arr_epoch_syn[i_synapse_threshold])

            min_idx = arr_energy_syn.index(min(arr_energy_syn))
            arr_energy[idx] = arr_energy_syn[min_idx]
            arr_energy_eLTP[idx] = arr_energy_eLTP_syn[min_idx]
            arr_energy_lLTP[idx] = arr_energy_lLTP_syn[min_idx]
            arr_error[idx] = arr_error_syn[min_idx]
            arr_epoch[idx] = arr_epoch_syn[min_idx]
            arr_opt_threshold[idx] = min_idx

            print("Optimal Threshold:", min_idx, " energy:", arr_energy[idx], " energy_eLTP:",
                  arr_energy_eLTP[idx], " energy_lLTP:",
                  arr_energy_lLTP[idx], " error:", arr_error[idx], " epoch:",
                  arr_epoch[idx])

        np.savetxt("../Text/Fig3/energy_" + str(i_maintenance_cost) + ".txt", arr_energy)
        np.savetxt("../Text/Fig3/energy_eLTP_" + str(i_maintenance_cost) + ".txt", arr_energy_eLTP)
        np.savetxt("../Text/Fig3/energy_lLTP_" + str(i_maintenance_cost) + ".txt", arr_energy_lLTP)
        np.savetxt("../Text/Fig3/error_" + str(i_maintenance_cost) + ".txt", arr_error)
        np.savetxt("../Text/Fig3/epoch_" + str(i_maintenance_cost) + ".txt", arr_epoch)
        np.savetxt("../Text/Fig3/opt_threshold_" + str(i_maintenance_cost) + ".txt", arr_opt_threshold)

    np.savetxt("../Text/decay_rates.txt", decay_rates)
    np.savetxt("../Text/maintenance_costs.txt", maintenance_costs)


transient_decay_rates()
np.savetxt("../Text/variables.txt", (
    nDimension, nPattern, nRun, learning_rate, energy_scale_lLTP, energy_scale_maintenance, synapse_threshold_max,
    decay_eLTP))
