#!/usr/bin/env python
import numpy as np
import MyFunction as MyF
from pathlib import Path
import multiprocessing as mp
import logging.handlers
import sys
import time

# Initialize logger
# logging.basicConfig(filename='../out.log', format='%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s',
#                     level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
log_formatter = logging.Formatter('%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s')
c_handler = logging.StreamHandler(sys.stdout)
c_handler.setFormatter(log_formatter)
f_handler = logging.handlers.RotatingFileHandler(filename='../out.log', maxBytes=(1048576*5), backupCount=7)
f_handler.setFormatter(log_formatter)
logger.addHandler(c_handler)
logger.addHandler(f_handler)


# Global variables
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
    decay_lLTP = 0.00001

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
            energy[iRun], energy_eLTP[iRun], energy_lLTP[iRun], error[iRun], epoch[iRun] = Per.AlgoStandard()
            logger.info(f"Run: {iRun} energy: {energy[iRun]} energy_eLTP: {energy_eLTP[iRun]} energy_lLTP: "
                        f"{energy_lLTP[iRun]} error: {error[iRun]} epoch: {epoch[iRun]}")

        arr_energy[idx] = np.mean(energy)
        arr_energy_eLTP[idx] = np.mean(energy_eLTP)
        arr_energy_lLTP[idx] = np.mean(energy_lLTP)
        arr_error[idx] = np.mean(error)
        arr_epoch[idx] = np.mean(epoch)
        logger.info(f"#Patterns: {nPatterns[idx]} energy: {arr_energy[idx]} energy_eLTP: {arr_energy_eLTP[idx]} "
                    f"energy_lLTP: {arr_energy_lLTP[idx]} error: {arr_error[idx]} epoch: {arr_epoch[idx]}")

    np.savetxt("../Text/Perm/energy.txt", arr_energy)
    np.savetxt("../Text/Perm/energy_eLTP.txt", arr_energy_eLTP)
    np.savetxt("../Text/Perm/energy_lLTP.txt", arr_energy_lLTP)
    np.savetxt("../Text/Perm/error.txt", arr_error)
    np.savetxt("../Text/Perm/epoch.txt", arr_epoch)

    np.savetxt("../Text/patterns.txt", nPatterns)
    np.savetxt("../Text/variables.txt", (
        nDimension, nRun, learning_rate, energy_scale_lLTP, energy_scale_maintenance, synapse_threshold_max,
        decay_eLTP))


nDimension = 1000
nPattern = 1600

decay_rates_lLTP = np.linspace(1e-6, 1e-4, 20)


def perm_decay_rates(iProcess):
    tmp = time.gmtime()
    np.random.seed(tmp[3] * (iProcess * 100 + tmp[4] * 10 + tmp[5]))

    arr_energy = np.nan * np.ones(len(decay_rates_lLTP))  # total energy
    arr_energy_eLTP = np.nan * np.ones(len(decay_rates_lLTP))  # transient energy
    arr_energy_lLTP = np.nan * np.ones(len(decay_rates_lLTP))  # permanent energy
    arr_error = np.nan * np.ones(len(decay_rates_lLTP))  # number of errors
    arr_epoch = np.nan * np.ones(len(decay_rates_lLTP))  # training time (epoch)
    arr_patterns = np.nan * np.ones(len(decay_rates_lLTP))  # min number of patterns that can be trained with zero error

    arr_std_epoch = np.nan * np.ones(len(decay_rates_lLTP))
    nPattern_prev = 0

    for idx in range(len(decay_rates_lLTP)):
        decay_lLTP = decay_rates_lLTP[idx]
        logger.info(f'Decay rate: {decay_lLTP}')
        iPattern = 100
        while iPattern <= nPattern:
            logger.info(f'Process: {iProcess} Trying for patterns: {iPattern}')
            weight_initial = np.zeros(nDimension + 1)  # +1 is for bias
            pattern, pattern_answer = create_patterns(iPattern, nDimension)
            Per = MyF.Perceptron(pattern[0], pattern_answer[0], weight_initial, size_buffer, learning_rate,
                                 energy_scale_lLTP, energy_detail=True)
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
                energy[iRun], energy_eLTP[iRun], energy_lLTP[iRun], error[iRun], epoch[iRun] = Per.AlgoStandard()
                logger.info(f"Process: {iProcess} Run: {iRun} energy: {energy[iRun]} energy_eLTP: {energy_eLTP[iRun]} "
                            f"energy_lLTP: {energy_lLTP[iRun]} error: {error[iRun]} epoch: {epoch[iRun]}")

            logger.info(f'Process: {iProcess} #Pattens: {iPattern} Mean Error: {np.mean(error)}')
            if np.mean(error) > 0.0:
                break

            mean_error = np.mean(error)
            mean_energy = np.mean(energy)
            mean_energy_eLTP = np.mean(energy_eLTP)
            mean_energy_lLTP = np.mean(energy_lLTP)
            mean_epoch = np.mean(epoch)
            std_epoch = np.std(epoch)

            nPattern_prev = iPattern
            iPattern += 20

        arr_energy[idx] = mean_energy
        arr_energy_eLTP[idx] = mean_energy_eLTP
        arr_energy_lLTP[idx] = mean_energy_lLTP
        arr_error[idx] = mean_error
        arr_epoch[idx] = mean_epoch
        arr_patterns[idx] = nPattern_prev

        arr_std_epoch[idx] = std_epoch
        logger.info(f'Process: {iProcess} #Patterns: {arr_patterns[idx]} energy: {arr_energy[idx]} energy_eLTP: '
                    f'{arr_energy_eLTP[idx]} energy_lLTP: {arr_energy_lLTP[idx]} error: {arr_error[idx]} epoch: '
                    f'{arr_epoch[idx]} std epoch: {arr_std_epoch[idx]}')

    Path("../Text/Perm_decay").mkdir(parents=True, exist_ok=True)
    np.savetxt("../Text/Perm_decay/energy_" + str(iProcess) + ".txt", arr_energy)
    np.savetxt("../Text/Perm_decay/energy_eLTP_" + str(iProcess) + ".txt", arr_energy_eLTP)
    np.savetxt("../Text/Perm_decay/energy_lLTP_" + str(iProcess) + ".txt", arr_energy_lLTP)
    np.savetxt("../Text/Perm_decay/error_" + str(iProcess) + ".txt", arr_error)
    np.savetxt("../Text/Perm_decay/epoch_" + str(iProcess) + ".txt", arr_epoch)
    np.savetxt("../Text/Perm_decay/patterns_" + str(iProcess) + ".txt", arr_patterns)
    np.savetxt("../Text/Perm_decay/std_epoch_" + str(iProcess) + ".txt", arr_std_epoch)


def combine_results(nProcess):
    # Combined mean and standard deviations
    arr_combined_mean_energy = np.nan * np.ones(len(decay_rates_lLTP))
    arr_combined_mean_error = np.nan * np.ones(len(decay_rates_lLTP))
    arr_combined_mean_epoch = np.nan * np.ones(len(decay_rates_lLTP))
    arr_combined_mean_patterns = np.nan * np.ones(len(decay_rates_lLTP))
    arr_combined_std_epoch = np.nan * np.ones(len(decay_rates_lLTP))

    # Arrays to store the values from all the processes
    arr_mean_energy = np.zeros(shape=(nProcess, len(decay_rates_lLTP)))
    arr_mean_error = np.zeros(shape=(nProcess, len(decay_rates_lLTP)))
    arr_mean_epoch = np.zeros(shape=(nProcess, len(decay_rates_lLTP)))
    arr_mean_patterns = np.zeros(shape=(nProcess, len(decay_rates_lLTP)))
    arr_std_epoch = np.zeros(shape=(nProcess, len(decay_rates_lLTP)))

    # Read the values of different processes from files.
    for iProcess in range(nProcess):
        arr_mean_energy[iProcess] = np.loadtxt("../Text/Perm_decay/energy_" + str(iProcess) + ".txt")
        arr_mean_error[iProcess] = np.loadtxt("../Text/Perm_decay/error_" + str(iProcess) + ".txt")
        arr_mean_epoch[iProcess] = np.loadtxt("../Text/Perm_decay/epoch_" + str(iProcess) + ".txt")
        arr_mean_patterns[iProcess] = np.loadtxt("../Text/Perm_decay/patterns_" + str(iProcess) + ".txt")
        arr_std_epoch[iProcess] = np.loadtxt("../Text/Perm_decay/std_epoch_" + str(iProcess) + ".txt")

    # Calculate the overall mean and standard deviation values
    for i in range(len(decay_rates_lLTP)):
        arr_combined_mean_energy[i] = np.mean(arr_mean_energy[:, i])
        arr_combined_mean_error[i] = np.mean(arr_mean_error[:, i])
        arr_combined_mean_epoch[i] = np.mean(arr_mean_epoch[:, i])
        arr_combined_mean_patterns[i] = np.mean(arr_mean_patterns[:, i])

        # Standard deviation
        std_sum = 0
        for j in range(nProcess):
            std_sum += arr_std_epoch[j][i] ** 2 + (arr_mean_epoch[j][i] - arr_combined_mean_epoch[i]) ** 2
        arr_combined_std_epoch[i] = np.sqrt(std_sum / nProcess)

    np.savetxt("../Text/Perm_decay/energy.txt", arr_combined_mean_energy)
    np.savetxt("../Text/Perm_decay/error.txt", arr_combined_mean_error)
    np.savetxt("../Text/Perm_decay/epoch.txt", arr_combined_mean_epoch)
    np.savetxt("../Text/Perm_decay/patterns.txt", arr_combined_mean_patterns)
    np.savetxt("../Text/Perm_decay/std_epoch.txt", arr_combined_std_epoch)


nProcess = 5
# pool = mp.Pool(nProcess)
# pool.map(perm_decay_rates, [iProcess for iProcess in range(nProcess)])
# pool.close()
# np.savetxt("../Text/Perm_decay/decay_rates.txt", decay_rates_lLTP)
# np.savetxt("../Text/Perm_decay/variables.txt", (nDimension, nPattern, nRun, learning_rate, energy_scale_lLTP,
#                                                 energy_scale_maintenance, synapse_threshold_max, decay_eLTP))
combine_results(nProcess)
