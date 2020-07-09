#!/usr/bin/env python
import numpy as np
import MyFunction as MyF
from pathlib import Path

nRun = 10  # one file contains events with one hyperplane solution
learning_rate = 1.0
size_buffer = 1000
energy_scale_lLTP = 1.0
energy_scale_maintenance = energy_scale_lLTP * np.array([0.001])
synapse_threshold = 2
decay_eLTP = 0.001
decay_lLTP = 0.00001

synapse_threshold_max = 40

#####################################
#        Pattern generation         #
#####################################


def create_patterns(nPattern, nDimension):
    pattern = np.array([None] * nRun)
    pattern_answer = np.array([None] * nRun)
    for iRun in range(nRun):
        pattern[iRun], pattern_answer[iRun] = MyF.MakePattern(nPattern, nDimension, is_pattern_integer=True,
                                                              is_plus_minus_one=True)
    return pattern, pattern_answer


#####################################
#          Perceptron part          #
#####################################

def perm_decay_patterns():
    nDimension = 1000
    nPatterns = np.arange(1000, 780, -20)  # Array of number of patterns

    arr_energy = np.nan * np.ones(len(nPatterns))  # total energy
    arr_energy_eLTP = np.nan * np.ones(len(nPatterns))  # transient energy
    arr_energy_lLTP = np.nan * np.ones(len(nPatterns))  # permanent energy
    arr_error = np.nan * np.ones(len(nPatterns))  # number of errors
    arr_epoch = np.nan * np.ones(len(nPatterns))  # training time (epoch)

    # Create a directory for output if it doesn't exist already
    Path("../Text/Perm").mkdir(parents=True, exist_ok=True)

    for idx in range(len(nPatterns)):
        weight_initial = np.zeros(nDimension + 1)  # +1 is for bias
        pattern, pattern_answer = create_patterns(nPatterns[idx], nDimension)
        Per = MyF.Perceptron(pattern[0], pattern_answer[0], weight_initial, \
                             size_buffer, learning_rate, energy_scale_lLTP, \
                             energy_detail=True)
        Per.energy_scale_maintenance = energy_scale_maintenance
        Per.decay_lLTP = decay_lLTP

        energy = np.nan * np.ones(nRun)
        energy_eLTP = np.nan * np.ones(nRun)
        energy_lLTP = np.nan * np.ones(nRun)
        error = np.nan * np.ones(nRun)
        epoch = np.nan * np.ones(nRun)
        for iRun in range(nRun):
            Per.pattern = pattern[iRun]
            Per.pattern_answer = pattern_answer[iRun]
            energy[iRun], energy_eLTP[iRun], energy_lLTP[iRun], error[iRun], epoch[iRun] = Per.AlgoSynapseAll(0)
            print("Run:", iRun+1, "energy:", energy[iRun], " energy_eLTP:", energy_eLTP[iRun], " energy_lLTP:",
                  energy_lLTP[iRun], " error:", error[iRun], " epoch:", epoch[iRun])

        arr_energy[idx] = np.mean(energy)
        arr_energy_eLTP[idx] = np.mean(energy_eLTP)
        arr_energy_lLTP[idx] = np.mean(energy_lLTP)
        arr_error[idx] = np.mean(error)
        arr_epoch[idx] = np.mean(epoch)
        print("#Patterns:", nPatterns[idx], " energy:", arr_energy[idx], " energy_eLTP:", arr_energy_eLTP[idx],
              "energy_lLTP:", arr_energy_lLTP[idx], " error:", arr_error[idx], " epoch:", arr_epoch[idx])

    np.savetxt("../Text/Perm/energy.txt", arr_energy)
    np.savetxt("../Text/Perm/energy_eLTP.txt", arr_energy_eLTP)
    np.savetxt("../Text/Perm/energy_lLTP.txt", arr_energy_lLTP)
    np.savetxt("../Text/Perm/error.txt", arr_error)
    np.savetxt("../Text/Perm/epoch.txt", arr_epoch)

    np.savetxt("../Text/patterns.txt", nPatterns)
    np.savetxt("../Text/variables.txt", (
        nDimension, nRun, learning_rate, energy_scale_lLTP, energy_scale_maintenance, synapse_threshold_max,
        decay_eLTP))


def perm_decay_rates():
    nDimension = 1000
    nPattern = 1000
    decay_rates_lLTP = np.arange(0.00001, 0.00005, 0.00001)

    arr_energy = np.nan * np.ones(len(decay_rates_lLTP))  # total energy
    arr_energy_eLTP = np.nan * np.ones(len(decay_rates_lLTP))  # transient energy
    arr_energy_lLTP = np.nan * np.ones(len(decay_rates_lLTP))  # permanent energy
    arr_error = np.nan * np.ones(len(decay_rates_lLTP))  # number of errors
    arr_epoch = np.nan * np.ones(len(decay_rates_lLTP))  # training time (epoch)
    arr_patterns = np.nan * np.ones(len(decay_rates_lLTP))  # min number of patterns that can be trained with zero error

    for idx in range(len(decay_rates_lLTP)):
        decay_lLTP = decay_rates_lLTP[idx]
        print('Decay rate:', decay_lLTP)
        iPattern = nPattern
        while iPattern > 0:
            print('Trying for patterns:', iPattern)
            weight_initial = np.zeros(nDimension + 1)  # +1 is for bias
            pattern, pattern_answer = create_patterns(iPattern, nDimension)
            Per = MyF.Perceptron(pattern[0], pattern_answer[0], weight_initial, \
                                 size_buffer, learning_rate, energy_scale_lLTP, \
                                 energy_detail=True)
            Per.energy_scale_maintenance = energy_scale_maintenance
            Per.decay_lLTP = decay_lLTP

            energy = np.nan * np.ones(nRun)
            energy_eLTP = np.nan * np.ones(nRun)
            energy_lLTP = np.nan * np.ones(nRun)
            error = np.nan * np.ones(nRun)
            epoch = np.nan * np.ones(nRun)
            for iRun in range(nRun):
                Per.pattern = pattern[iRun]
                Per.pattern_answer = pattern_answer[iRun]
                energy[iRun], energy_eLTP[iRun], energy_lLTP[iRun], error[iRun], epoch[iRun] = Per.AlgoSynapseAll(0)
                print("Run:", iRun + 1, "energy:", energy[iRun], " energy_eLTP:", energy_eLTP[iRun], " energy_lLTP:",
                      energy_lLTP[iRun], " error:", error[iRun], " epoch:", epoch[iRun])

            mean_error = np.mean(error)
            mean_energy = np.mean(energy)
            mean_energy_eLTP = np.mean(energy_eLTP)
            mean_energy_lLTP = np.mean(energy_lLTP)
            mean_epoch = np.mean(epoch)
            print("#Pattens:", iPattern, "Mean Error:", mean_error)

            if mean_error == 0.0:
                break

            iPattern -= 20


        arr_energy[idx] = mean_energy
        arr_energy_eLTP[idx] = mean_energy_eLTP
        arr_energy_lLTP[idx] = mean_energy_lLTP
        arr_error[idx] = mean_error
        arr_epoch[idx] = mean_epoch
        arr_patterns[idx] = iPattern
        print("#Patterns:", arr_patterns[idx], " energy:", arr_energy[idx], " energy_eLTP:", arr_energy_eLTP[idx],
              "energy_lLTP:", arr_energy_lLTP[idx], " error:", arr_error[idx], " epoch:", arr_epoch[idx])

    Path("../Text/Perm_decay").mkdir(parents=True, exist_ok=True)
    np.savetxt("../Text/Perm_decay/energy.txt", arr_energy)
    np.savetxt("../Text/Perm_decay/energy_eLTP.txt", arr_energy_eLTP)
    np.savetxt("../Text/Perm_decay/energy_lLTP.txt", arr_energy_lLTP)
    np.savetxt("../Text/Perm_decay/error.txt", arr_error)
    np.savetxt("../Text/Perm_decay/epoch.txt", arr_epoch)
    np.savetxt("../Text/Perm_decay/patterns.txt", arr_patterns)

    np.savetxt("../Text/Perm_decay/decay_rates.txt", decay_rates_lLTP)
    np.savetxt("../Text/Perm_decay/variables.txt", (
        nDimension, nPattern, nRun, learning_rate, energy_scale_lLTP, energy_scale_maintenance, synapse_threshold_max,
        decay_eLTP))


# perm_decay_patterns()
perm_decay_rates()
