#!/usr/bin/env python
import numpy as np
import MyFunction as MyF

nDimension = 1000
nPattern = 1000
nRun = 1  # one file contains events with one hyperplane solution
learning_rate = 1.0
size_buffer = 1000
energy_scale_lLTP = 1.0
energy_scale_maintenance = energy_scale_lLTP * np.array([0.001])
synapse_threshold = 2
arr_decay_eLTP = 0.001

synapse_threshold_max = 40
decay_rates = np.arange(0, 0.005, 0.001)
learning_rates = np.array([0.1, 0.5, 1.0, 1.5, 2.0])

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
weight_initial = np.zeros(nDimension + 1)  # +1 is for bias
Per = MyF.Perceptron(pattern[0], pattern_answer[0], weight_initial, \
                     size_buffer, learning_rate, energy_scale_lLTP, \
                     energy_detail=True)
Per.energy_scale_maintenance = energy_scale_maintenance
# arr_energy = np.nan*np.ones(nRun) # total energy
# arr_energy_eLTP = np.nan*np.ones(nRun) # transient energy
# arr_energy_lLTP = np.nan*np.ones(nRun) # permanent energy
# arr_error = np.nan*np.ones(nRun) # number of errors
# arr_epoch = np.nan*np.ones(nRun) # training time (epoch)

arr_energy = np.nan * np.ones(synapse_threshold_max)  # total energy
arr_energy_eLTP = np.nan * np.ones(synapse_threshold_max)  # transient energy
arr_energy_lLTP = np.nan * np.ones(synapse_threshold_max)  # permanent energy
arr_error = np.nan * np.ones(synapse_threshold_max)  # number of errors
arr_epoch = np.nan * np.ones(synapse_threshold_max)  # training time (epoch)

for decay_rate in decay_rates:
    Per.decay_eLTP = decay_rate
    for iRun in range(synapse_threshold_max):
        Per.pattern = pattern[0]
        Per.pattern_answer = pattern_answer[0]
        arr_energy[iRun], arr_energy_eLTP[iRun], arr_energy_lLTP[iRun], arr_error[iRun], arr_epoch[
            iRun] = Per.AlgoSynapseAll(iRun)
        print(iRun, " energy:", arr_energy[iRun], " energy_eLTP:", arr_energy_eLTP[iRun], " energy_lLTP:",
              arr_energy_lLTP[iRun], " error:", arr_error[iRun], " epoch:", arr_epoch[iRun])

    np.savetxt("Text/energy_" + str(decay_rate) + ".txt", arr_energy)
    np.savetxt("Text/energy_eLTP_" + str(decay_rate) + ".txt", arr_energy_eLTP)
    np.savetxt("Text/energy_lLTP_" + str(decay_rate) + ".txt", arr_energy_lLTP)
    np.savetxt("Text/error_" + str(decay_rate) + ".txt", arr_error)
    np.savetxt("Text/epoch_" + str(decay_rate) + ".txt", arr_epoch)

# Per.decay_eLTP = 0.001
# for learning_rate in learning_rates:
#     Per.learning_rate = learning_rate
#     for iRun in range(synapse_threshold_max):
#         Per.pattern = pattern[0]
#         Per.pattern_answer = pattern_answer[0]
#         arr_energy[iRun], arr_energy_eLTP[iRun], arr_energy_lLTP[iRun], arr_error[iRun], arr_epoch[
#             iRun] = Per.AlgoSynapseAll(iRun)
#         print(iRun, " energy:", arr_energy[iRun], " energy_eLTP:", arr_energy_eLTP[iRun], " energy_lLTP:",
#               arr_energy_lLTP[iRun], " error:", arr_error[iRun], " epoch:", arr_epoch[iRun])
#     np.savetxt("Text/energy_" + str(learning_rate) + ".txt", arr_energy)
#     np.savetxt("Text/energy_eLTP_" + str(learning_rate) + ".txt", arr_energy_eLTP)
#     np.savetxt("Text/energy_lLTP_" + str(learning_rate) + ".txt", arr_energy_lLTP)
#     np.savetxt("Text/error_" + str(learning_rate) + ".txt", arr_error)
#     np.savetxt("Text/epoch_" + str(learning_rate) + ".txt", arr_epoch)

np.savetxt("Text/variables.txt", (
nDimension, nPattern, nRun, learning_rate, energy_scale_lLTP, energy_scale_maintenance, synapse_threshold_max,
arr_decay_eLTP))
np.savetxt("Text/decay_rates.txt", decay_rates)
np.savetxt("Text/learning_rates.txt", learning_rates)
