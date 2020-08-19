#!/usr/bin/env python
from collections import OrderedDict

import numpy as np
from pathlib import Path
import multiprocessing as mp
import time
from functools import partial
import copy
import matplotlib.pyplot as plt

import Code.Perceptron as Perceptron
import Code.log_helper as log_helper
import Code.MyConstants as Constants
from Code.Forgetting import Forgetting

logger = log_helper.get_logger('perceptron_helper')

# Global variables
learning_rate = 1.0
size_buffer = 1000
energy_scale_lLTP = 1.0
energy_scale_maintenance = energy_scale_lLTP * np.array([0.001])
decay_eLTP = 0.001


#####################################
#        Pattern generation         #
#####################################


def create_patterns(nPattern, nDimension, nRun):
    pattern = np.array([None] * nRun)
    pattern_answer = np.array([None] * nRun)
    for iRun in range(nRun):
        pattern[iRun], pattern_answer[iRun] = Perceptron.MakePattern(nPattern, nDimension, is_pattern_integer=True,
                                                                     is_plus_minus_one=True)
    return pattern, pattern_answer


#####################################
#          Perceptron part          #
#####################################

def perm_decay_patterns():
    nRun = 10
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
        pattern, pattern_answer = create_patterns(nPatterns[idx], nDimension, nRun)
        Per = Perceptron.Perceptron(pattern[0], pattern_answer[0], weight_initial, \
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
        nDimension, nRun, learning_rate, energy_scale_lLTP, energy_scale_maintenance, decay_eLTP))


def perm_decay_rates(iProcess, **kwargs):
    # New random seed for each process
    tmp = time.gmtime()
    np.random.seed(tmp[3] * (iProcess * 100 + tmp[4] * 10 + tmp[5]))

    # Read all the arguments
    nDimension = kwargs['nDimension']
    nPattern = kwargs['nPattern']
    decay_rates_lLTP = kwargs['decay_rates_lLTP']
    iPattern_init = kwargs['iPattern_init']
    step_size = kwargs['step_size']
    output_path = kwargs['output_path']
    nRun = kwargs['nRun']

    # Arrays to store the mean/std values of different measurements from all the runs
    arr_mean_energy = np.nan * np.ones(len(decay_rates_lLTP))  # total energy
    arr_mean_energy_eLTP = np.nan * np.ones(len(decay_rates_lLTP))  # transient energy
    arr_mean_energy_lLTP = np.nan * np.ones(len(decay_rates_lLTP))  # permanent energy
    arr_mean_error = np.nan * np.ones(len(decay_rates_lLTP))  # number of errors
    arr_mean_epoch = np.nan * np.ones(len(decay_rates_lLTP))  # training time (epoch)
    arr_mean_patterns = np.nan * np.ones(
        len(decay_rates_lLTP))  # max number of patterns that can be trained with zero error
    arr_std_epoch = np.nan * np.ones(len(decay_rates_lLTP))

    # Arrays to store the individual values of all the measurements
    arr_all_energy = np.nan * np.ones(shape=(len(decay_rates_lLTP), nRun))
    arr_all_energy_eLTP = np.nan * np.ones(shape=(len(decay_rates_lLTP), nRun))
    arr_all_energy_lLTP = np.nan * np.ones(shape=(len(decay_rates_lLTP), nRun))
    arr_all_error = np.nan * np.ones(shape=(len(decay_rates_lLTP), nRun))
    arr_all_epoch = np.nan * np.ones(shape=(len(decay_rates_lLTP), nRun))

    nPattern_prev = 0

    for idx in range(len(decay_rates_lLTP)):
        decay_lLTP = decay_rates_lLTP[idx]
        logger.info(f'Decay rate: {decay_lLTP}')
        iPattern = iPattern_init
        while iPattern <= nPattern:
            logger.info(f'Process: {iProcess} Trying for patterns: {iPattern}')
            weight_initial = np.zeros(nDimension + 1)  # +1 is for bias
            pattern, pattern_answer = create_patterns(iPattern, nDimension, nRun)
            Per = Perceptron.Perceptron(pattern[0], pattern_answer[0], weight_initial, size_buffer, learning_rate,
                                        energy_scale_lLTP, energy_detail=True, use_accuracy=False)
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

            # Copy values of all the runs to the 2D array
            arr_all_energy[idx] = energy
            arr_all_energy_eLTP[idx] = energy_eLTP
            arr_all_energy_lLTP[idx] = energy_lLTP
            arr_all_error[idx] = error
            arr_all_epoch[idx] = epoch

            # Calculate the mean and std of all the runs
            mean_error = np.mean(error)
            mean_energy = np.mean(energy)
            mean_energy_eLTP = np.mean(energy_eLTP)
            mean_energy_lLTP = np.mean(energy_lLTP)
            mean_epoch = np.mean(epoch)
            std_epoch = np.std(epoch)

            nPattern_prev = iPattern
            iPattern += step_size

        arr_mean_energy[idx] = mean_energy
        arr_mean_energy_eLTP[idx] = mean_energy_eLTP
        arr_mean_energy_lLTP[idx] = mean_energy_lLTP
        arr_mean_error[idx] = mean_error
        arr_mean_epoch[idx] = mean_epoch
        arr_mean_patterns[idx] = nPattern_prev

        arr_std_epoch[idx] = std_epoch
        logger.info(
            f'Process: {iProcess} #Patterns: {arr_mean_patterns[idx]} energy: {arr_mean_energy[idx]} energy_eLTP: '
            f'{arr_mean_energy_eLTP[idx]} energy_lLTP: {arr_mean_energy_lLTP[idx]} error: {arr_mean_error[idx]} epoch: '
            f'{arr_mean_epoch[idx]} std epoch: {arr_std_epoch[idx]}')

    Path(output_path).mkdir(parents=True, exist_ok=True)
    # Write all the 2D arrays (values from all the runs) to files
    np.savetxt(output_path + Constants.ENERGY_FILE_PROC_ALL.format(str(iProcess)), arr_all_energy)
    np.savetxt(output_path + Constants.ENERGY_ELTP_FILE_PROC_ALL.format(str(iProcess)), arr_all_energy_eLTP)
    np.savetxt(output_path + Constants.ENERGY_LLTP_FILE_PROC_ALL.format(str(iProcess)), arr_all_energy_lLTP)
    np.savetxt(output_path + Constants.ERROR_FILE_PROC_ALL.format(str(iProcess)), arr_all_error)
    np.savetxt(output_path + Constants.EPOCH_FILE_PROC_ALL.format(str(iProcess)), arr_all_epoch)

    # Write the mean/std values of the runs to files
    np.savetxt(output_path + Constants.ENERGY_FILE_PROC.format(str(iProcess)), arr_mean_energy)
    np.savetxt(output_path + Constants.ENERGY_ELTP_FILE_PROC.format(str(iProcess)), arr_mean_energy_eLTP)
    np.savetxt(output_path + Constants.ENERGY_LLTP_FILE_PROC.format(str(iProcess)), arr_mean_energy_lLTP)
    np.savetxt(output_path + Constants.ERROR_FILE_PROC.format(str(iProcess)), arr_mean_error)
    np.savetxt(output_path + Constants.EPOCH_FILE_PROC.format(str(iProcess)), arr_mean_epoch)
    np.savetxt(output_path + Constants.PATTERNS_FILE_PROC.format(str(iProcess)), arr_mean_patterns)
    np.savetxt(output_path + Constants.STD_EPOCH_FILE_PROC.format(str(iProcess)), arr_std_epoch)


def combine_perm_decay_results(**kwargs):
    decay_rates_lLTP = kwargs['decay_rates_lLTP']
    output_path = kwargs['output_path']
    nProcess = kwargs['nProcess']

    # Combined mean and standard deviations
    arr_combined_mean_energy = np.nan * np.ones(len(decay_rates_lLTP))
    arr_combined_mean_error = np.nan * np.ones(len(decay_rates_lLTP))
    arr_combined_mean_epoch = np.nan * np.ones(len(decay_rates_lLTP))
    arr_combined_mean_patterns = np.nan * np.ones(len(decay_rates_lLTP))
    arr_combined_std_epoch = np.nan * np.ones(len(decay_rates_lLTP))
    arr_combined_std_energy = np.nan * np.ones(len(decay_rates_lLTP))
    arr_combined_std_patterns = np.nan * np.ones(len(decay_rates_lLTP))

    # Arrays to store the mean/std values from all the processes
    arr_mean_energy = np.zeros(shape=(nProcess, len(decay_rates_lLTP)))
    arr_mean_error = np.zeros(shape=(nProcess, len(decay_rates_lLTP)))
    arr_mean_epoch = np.zeros(shape=(nProcess, len(decay_rates_lLTP)))
    arr_mean_patterns = np.zeros(shape=(nProcess, len(decay_rates_lLTP)))
    arr_std_epoch = np.zeros(shape=(nProcess, len(decay_rates_lLTP)))

    # Arrays to store all the run values from all the processes
    arr_all_energy = None
    arr_all_epoch = None

    # Read the values of different processes from files.
    for iProcess in range(nProcess):
        arr_mean_energy[iProcess] = np.loadtxt(output_path + Constants.ENERGY_FILE_PROC.format(str(iProcess)))
        arr_mean_error[iProcess] = np.loadtxt(output_path + Constants.ERROR_FILE_PROC.format(str(iProcess)))
        arr_mean_epoch[iProcess] = np.loadtxt(output_path + Constants.EPOCH_FILE_PROC.format(str(iProcess)))
        arr_mean_patterns[iProcess] = np.loadtxt(output_path + Constants.PATTERNS_FILE_PROC.format(str(iProcess)))
        arr_std_epoch[iProcess] = np.loadtxt(output_path + Constants.STD_EPOCH_FILE_PROC.format(str(iProcess)))

        # Read values from all the runs
        new_arr_energy = np.loadtxt(output_path + Constants.ENERGY_FILE_PROC_ALL.format(str(iProcess)))
        new_arr_epoch = np.loadtxt(output_path + Constants.EPOCH_FILE_PROC_ALL.format(str(iProcess)))
        if arr_all_energy is None:
            arr_all_energy = new_arr_energy
        else:
            arr_all_energy = np.append(arr_all_energy, new_arr_energy, axis=1)

        if arr_all_epoch is None:
            arr_all_epoch = new_arr_epoch
        else:
            arr_all_epoch = np.append(arr_all_epoch, new_arr_epoch, axis=1)

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

        arr_combined_std_energy[i] = np.std(arr_all_energy[i, :])
        arr_combined_std_patterns[i] = np.std(arr_mean_patterns[:, i])

    np.savetxt(output_path + Constants.ENERGY_FILE, arr_combined_mean_energy)
    np.savetxt(output_path + Constants.ERROR_FILE, arr_combined_mean_error)
    np.savetxt(output_path + Constants.EPOCH_FILE, arr_combined_mean_epoch)
    np.savetxt(output_path + Constants.PATTERNS_FILE, arr_combined_mean_patterns)
    np.savetxt(output_path + Constants.STD_EPOCH_FILE, arr_combined_std_epoch)
    np.savetxt(output_path + Constants.STD_ENERGY_FILE, arr_combined_std_energy)
    np.savetxt(output_path + Constants.STD_PATTERNS_FILE, arr_combined_std_patterns)


def perceptron_accuracy(iProcess, **kwargs):
    # New random seed for each process
    tmp = time.gmtime()
    np.random.seed(tmp[3] * (iProcess * 100 + tmp[4] * 10 + tmp[5]))

    # Read all the arguments
    nDimension = kwargs['nDimension']
    nPattern = kwargs['nPattern']
    decay_rates_lLTP = kwargs['decay_rates_lLTP']
    output_path = kwargs['output_path']
    nRun = kwargs['nRun']
    epoch_updates = kwargs['epoch_updates']
    Path(output_path).mkdir(parents=True, exist_ok=True)

    window_size = kwargs['window_size']

    weight_initial = np.zeros(nDimension + 1)  # +1 is for bias
    pattern, pattern_answer = create_patterns(nPattern, nDimension, nRun)
    Per = Perceptron.Perceptron(pattern[0], pattern_answer[0], weight_initial, size_buffer, learning_rate,
                                energy_scale_lLTP, energy_detail=True, use_accuracy=True)
    Per.energy_scale_maintenance = energy_scale_maintenance

    arr_mean_accuracy = np.nan * np.ones(shape=(len(decay_rates_lLTP)))
    arr_mean_error = np.nan * np.ones(shape=(len(decay_rates_lLTP)))

    # To store the values of all runs
    arr_all_energy = np.nan * np.ones(shape=(len(decay_rates_lLTP), nRun))
    arr_all_error = np.nan * np.ones(shape=(len(decay_rates_lLTP), nRun))
    arr_all_epoch = np.nan * np.ones(shape=(len(decay_rates_lLTP), nRun))
    arr_all_accuracy = np.nan * np.ones(shape=(len(decay_rates_lLTP), nRun))

    if epoch_updates:
        arr_all_epoch_updates = []
        arr_all_energy_updates = []

    for j in range(len(decay_rates_lLTP)):
        decay_lLTP = decay_rates_lLTP[j]
        logger.info(f'Decay rate: {decay_lLTP}')
        Per.decay_lLTP = decay_lLTP
        Per.epoch_window = window_size

        energy = np.nan * np.ones(nRun)
        energy_eLTP = np.nan * np.ones(nRun)
        energy_lLTP = np.nan * np.ones(nRun)
        error = np.nan * np.ones(nRun)
        epoch = np.nan * np.ones(nRun)
        accuracy = np.nan * np.ones(nRun)
        epoch_updates = [[] for i in range(nRun)]
        energy_updates = [[] for i in range(nRun)]
        for iRun in range(nRun):
            Per.pattern = pattern[iRun]
            Per.pattern_answer = pattern_answer[iRun]
            energy[iRun], energy_eLTP[iRun], energy_lLTP[iRun], error[iRun], epoch[iRun], accuracy[iRun], \
                epoch_updates[iRun], energy_updates[iRun] = Per.AlgoStandard()
            logger.info(f"Process: {iProcess} Run: {iRun} energy: {energy[iRun]} energy_eLTP: {energy_eLTP[iRun]} "
                        f"energy_lLTP: {energy_lLTP[iRun]} error: {error[iRun]} epoch: {epoch[iRun]} "
                        f"accuracy: {accuracy[iRun]}")

        arr_all_energy[j] = energy
        arr_all_error[j] = error
        arr_all_epoch[j] = epoch
        arr_all_accuracy[j] = accuracy

        if epoch_updates:
            arr_all_epoch_updates.append(epoch_updates)
            arr_all_energy_updates.append(energy_updates)

        arr_mean_accuracy[j] = np.mean(accuracy)
        arr_mean_error[j] = np.mean(error)

    np.savetxt(output_path + Constants.ACCURACY_FILE_PROC.format(str(iProcess)), arr_mean_accuracy)
    np.savetxt(output_path + Constants.ERROR_FILE_PROC.format(str(iProcess)), arr_mean_error)

    np.savetxt(output_path + Constants.ACCURACY_FILE_PROC_ALL.format((str(iProcess))), arr_all_accuracy)
    np.savetxt(output_path + Constants.ERROR_FILE_PROC_ALL.format((str(iProcess))), arr_all_error)
    np.savetxt(output_path + Constants.ENERGY_FILE_PROC_ALL.format((str(iProcess))), arr_all_energy)
    np.savetxt(output_path + Constants.EPOCH_FILE_PROC_ALL.format((str(iProcess))), arr_all_epoch)

    if epoch_updates:
        np.save(output_path + Constants.EPOCH_UPDATES_ALL, np.array(arr_all_epoch_updates, dtype=object),
                allow_pickle=True)
        np.save(output_path + Constants.ENERGY_UPDATES_ALL, np.array(arr_all_energy_updates,
                                                                     dtype=object), allow_pickle=True)


def perceptron_forget_and_learn(Per, nDimension, n_iter, new_patterns_count):
    # Generate new patterns
    start_index = 0
    end_index = new_patterns_count
    # n_iter = int(nPattern / new_patterns_count)
    f_energy = np.nan * np.ones(n_iter)
    f_energy_eLTP = np.nan * np.ones(n_iter)
    f_energy_lLTP = np.nan * np.ones(n_iter)
    f_error = np.nan * np.ones(n_iter)
    f_epoch = np.nan * np.ones(n_iter)
    f_accuracy = np.nan * np.ones(n_iter)

    f_epoch_updates = [[] for i in range(n_iter)]
    f_energy_updates = [[] for i in range(n_iter)]

    f_epoch_updates_np = [[] for i in range(n_iter)]

    idx = 0
    while idx < n_iter:
        # logger.info(f'Start index: {start_index} End index: {end_index}')
        pattern, pattern_answer = create_patterns(new_patterns_count, nDimension, 1)
        Per.pattern[start_index:end_index, :] = pattern[0]
        Per.pattern_answer[start_index:end_index] = pattern_answer[0]

        # Train the perceptron with the new set of patterns
        f_energy[idx], f_energy_eLTP[idx], f_energy_lLTP[idx], f_error[idx], f_epoch[idx], f_accuracy[idx], \
            f_epoch_updates[idx], f_energy_updates[idx] = Per.AlgoStandard(start_index, end_index)
        f_epoch_updates_np[idx] = Per.arr_epoch_updates_np

        logger.info(f"Iteration: {idx} energy: {f_energy[idx]} energy_eLTP: {f_energy_eLTP[idx]} "
                    f"energy_lLTP: {f_energy_lLTP[idx]} error: {f_error[idx]} epoch: {f_epoch[idx]} "
                    f"accuracy: {f_accuracy[idx]}")
        # logger.info(f'||W||: {np.linalg.norm(Per.arr_weight)}')

        start_index = end_index
        end_index += new_patterns_count
        idx += 1

    return f_accuracy, f_energy, f_epoch_updates, f_energy_updates, f_epoch_updates_np


def perceptron_forgetting_benchmark(iProcess, iRun, nDimension, new_patterns_count, Per):
    # Train the perceptron with x new patterns
    Per.initialize_weights = False
    pattern, pattern_answer = create_patterns(new_patterns_count, nDimension, 1)
    Per.pattern = np.concatenate((Per.pattern, pattern[0]))
    Per.pattern_answer = np.concatenate((Per.pattern_answer, pattern_answer[0]))
    Per.nPattern = len(Per.pattern_answer)

    energy, energy_eLTP, energy_lLTP, error, epoch, accuracy, epoch_updates, energy_updates = Per.AlgoStandard()
    logger.info(
        f"Process: {iProcess} perceptron_forgetting_benchmark: Run: {iRun} energy: {energy} accuracy: {accuracy}")

    return accuracy, energy


def perceptron_cat_forgetting_1(iProcess, iRun, nDimension, new_patterns_count, Per):
    # Randomly select y old patterns and add x new patterns to train the perceptron
    Per.initialize_weights = False

    # Random selection of patterns from the original pattern array
    rnd_count = 700
    index = np.random.choice(Per.pattern.shape[0], rnd_count, replace=False)
    new_patterns = Per.pattern[index]
    new_pattern_answers = Per.pattern_answer[index]

    # Generate new patterns
    pattern, pattern_answer = create_patterns(new_patterns_count, nDimension, 1)

    # Concatenate the randomly chosen patterns with the new patterns
    Per.pattern = np.concatenate((new_patterns, pattern[0]))
    Per.pattern_answer = np.concatenate((new_pattern_answers, pattern_answer[0]))
    Per.nPattern = len(Per.pattern_answer)

    energy, energy_eLTP, energy_lLTP, error, epoch, accuracy, epoch_updates, energy_updates = Per.AlgoStandard()
    logger.info(
        f"Process: {iProcess} perceptron_cat_forgetting_1: Run: {iRun} energy: {energy} accuracy: {accuracy}")

    return accuracy, energy


def perceptron_cat_forgetting_2(iProcess, iRun, nDimension, new_patterns_count, Per):
    Per.initialize_weights = False

    # Generate new patterns
    pattern, pattern_answer = create_patterns(new_patterns_count, nDimension, 1)
    Per.pattern = pattern[0]
    Per.pattern_answer = pattern_answer[0]
    Per.nPattern = len(Per.pattern_answer)

    energy, energy_eLTP, energy_lLTP, error, epoch, accuracy, epoch_updates, energy_updates = Per.AlgoStandard()
    logger.info(
        f"Process: {iProcess} perceptron_cat_forgetting_2: Run: {iRun} energy: {energy} accuracy: {accuracy}")

    return accuracy, energy


def perceptron_cat_forgetting_3(iProcess, iRun, nDimension, n_iter, new_patterns_count, Per):
    # Train the perceptron without any decay and learn new patterns
    Per.initialize_weights = False
    accuracy, energy, epoch_updates, energy_updates, epoch_updates_np = \
        perceptron_forget_and_learn(Per, nDimension, n_iter, new_patterns_count)
    logger.info(f"Process: {iProcess} perceptron_cat_forgetting_3: Run: {iRun} energy: {energy} accuracy: {accuracy}")

    # if iRun == 0 and iProcess == 0:
    #     np.save(output_path + Constants.EPOCH_UPDATES_WITHOUT_DECAY_ALL, np.array(epoch_updates, dtype=object),
    #             allow_pickle=True)
    #     np.save(output_path + Constants.EPOCH_UPDATES_NP_WITHOUT_DECAY_ALL, np.array(epoch_updates_np, dtype=object),
    #             allow_pickle=True)
    #     np.save(output_path + Constants.ENERGY_UPDATES_WITHOUT_DECAY_ALL, np.array(energy_updates, dtype=object),
    #             allow_pickle=True)

    return accuracy, energy


def perceptron_active_forgetting_1(iProcess, iRun, nDimension, n_iter, new_patterns_count, decay_rate, Per):
    Per.initialize_weights = False
    Per.decay_lLTP = decay_rate

    # Generate new patterns
    pattern, pattern_answer = create_patterns(new_patterns_count, nDimension, 1)
    Per.pattern = np.concatenate((Per.pattern, pattern[0]))
    Per.pattern_answer = np.concatenate((Per.pattern_answer, pattern_answer[0]))
    Per.nPattern = len(Per.pattern_answer)

    energy, energy_eLTP, energy_lLTP, error, epoch, accuracy, epoch_updates, energy_updates = Per.AlgoStandard()
    logger.info(
        f"Process: {iProcess} perceptron_active_forgetting_1: Run: {iRun} energy: {energy} accuracy: {accuracy}")

    return accuracy, energy


def perceptron_active_forgetting_2(iProcess, iRun, nDimension, n_iter, new_patterns_count, decay_rate, Per):
    Per.initialize_weights = False
    Per.decay_lLTP = decay_rate
    logger.info(f'Decay rate: {Per.decay_lLTP}')
    accuracy, energy, epoch_updates, energy_updates, epoch_updates_np = \
        perceptron_forget_and_learn(Per, nDimension, n_iter, new_patterns_count)
    logger.info(f"Process: {iProcess} WITH DECAY: Run: {iRun} energy: {energy} accuracy: {accuracy}")

    # if iRun == 0 and iProcess == 0:
    #     np.save(output_path + Constants.EPOCH_UPDATES_ALL, np.array(epoch_updates, dtype=object),
    #             allow_pickle=True)
    #     np.save(output_path + Constants.EPOCH_UPDATES_NP_ALL, np.array(epoch_updates_np, dtype=object),
    #             allow_pickle=True)
    #     np.save(output_path + Constants.ENERGY_UPDATES_ALL, np.array(energy_updates, dtype=object),
    #             allow_pickle=True)

    return accuracy, energy


def perceptron_forgetting(iProcess, **kwargs):
    # New random seed for each process
    tmp = time.gmtime()
    np.random.seed(tmp[3] * (iProcess * 100 + tmp[4] * 10 + tmp[5]))

    # Read all the arguments
    nDimension = kwargs['nDimension']
    nPattern = kwargs['nPattern']
    output_path = kwargs['output_path']
    nRun = kwargs['nRun']
    window_size = kwargs['window_size']
    new_patterns_count = kwargs['new_patterns_count']
    n_iter = kwargs['n_iter']
    decay_rate = kwargs['decay_rate']

    weight_initial = np.zeros(nDimension + 1)  # +1 is for bias
    pattern, pattern_answer = create_patterns(nPattern, nDimension, nRun)

    # Benchmark
    benchmark_forgetting = Forgetting(nRun, output_path + Constants.BENCHMARK_FORGETTING)

    # Catastrophic forgetting 1
    cat_forgetting_1 = Forgetting(nRun, output_path + Constants.CAT_FORGETTING_1)

    # Catastrophic forgetting 2
    cat_forgetting_2 = Forgetting(nRun, output_path + Constants.CAT_FORGETTING_2)

    # Active forgetting 1
    active_forgetting_1 = Forgetting(nRun, output_path + Constants.ACTIVE_FORGETTING_1)

    # Active forgetting 2
    active_forgetting_2 = Forgetting(nRun, output_path + Constants.ACTIVE_FORGETTING_2, n_iter)

    # Catastrophic forgetting 3
    cat_forgetting_3 = Forgetting(nRun, output_path + Constants.CAT_FORGETTING_3, n_iter)

    for iRun in range(nRun):
        ####### Base training ######
        # Train the perceptron without any decay
        Per = Perceptron.Perceptron(pattern[iRun], pattern_answer[iRun], weight_initial, size_buffer, learning_rate,
                                    energy_scale_lLTP, energy_detail=True, use_accuracy=True)
        Per.energy_scale_maintenance = energy_scale_maintenance
        Per.epoch_window = window_size

        Per.decay_lLTP = 0.0
        energy, energy_eLTP, energy_lLTP, error, epoch, accuracy, epoch_updates, energy_updates = Per.AlgoStandard()
        logger.info(
            f"Process: {iProcess} energy: {energy} energy_eLTP: {energy_eLTP} energy_lLTP: {energy_lLTP} error: {error}"
            f" epoch: {epoch} accuracy: {accuracy}")

        # Make a copy of the object to use it later
        base_per_obj = copy.deepcopy(Per)

        #### 1. Benchmark ####
        accuracy, energy = perceptron_forgetting_benchmark(iProcess, iRun, nDimension, new_patterns_count, Per)
        benchmark_forgetting.energy[iRun] = energy
        benchmark_forgetting.accuracy[iRun] = accuracy

        #### 2. Catastrophic forgetting - 1 ####
        Per = copy.deepcopy(base_per_obj)
        accuracy, energy = perceptron_cat_forgetting_1(iProcess, iRun, nDimension, new_patterns_count, Per)
        cat_forgetting_1.accuracy[iRun] = accuracy
        cat_forgetting_1.energy[iRun] = energy

        #### 3. Catastrophic forgetting - 2 ####
        Per = copy.deepcopy(base_per_obj)
        accuracy, energy = perceptron_cat_forgetting_2(iProcess, iRun, nDimension, new_patterns_count, Per)
        cat_forgetting_2.accuracy[iRun] = accuracy
        cat_forgetting_2.energy[iRun] = energy

        #### 4. Active forgetting - 1 ####
        Per = copy.deepcopy(base_per_obj)
        accuracy, energy = perceptron_active_forgetting_1(iProcess, iRun, nDimension, n_iter, new_patterns_count,
                                                          decay_rate, Per)
        active_forgetting_1.accuracy[iRun] = accuracy
        active_forgetting_1.energy[iRun] = energy

        #### 5. Catastrophic forgetting - 3 ####
        # Train the perceptron without any decay and learn new patterns
        Per = copy.deepcopy(base_per_obj)
        accuracy, energy = perceptron_cat_forgetting_3(iProcess, iRun, nDimension, n_iter, new_patterns_count,
                                                       Per)
        cat_forgetting_3.accuracy[iRun] = accuracy
        cat_forgetting_3.energy[iRun] = energy

        #### 6. Active forgetting - 2 ####
        # Add decay to the perceptron and learn new patterns
        Per = copy.deepcopy(base_per_obj)
        accuracy, energy = perceptron_active_forgetting_2(iProcess, iRun, nDimension, n_iter, new_patterns_count,
                                                          decay_rate, Per)
        active_forgetting_2.accuracy[iRun] = accuracy
        active_forgetting_2.energy[iRun] = energy

    # Save all the output values to files.
    benchmark_forgetting.save_proc_output(iProcess)
    cat_forgetting_1.save_proc_output(iProcess)
    cat_forgetting_2.save_proc_output(iProcess)
    cat_forgetting_3.save_proc_output(iProcess)
    active_forgetting_1.save_proc_output(iProcess)
    active_forgetting_2.save_proc_output(iProcess)


def combine_perceptron_forgetting_results(**kwargs):
    output_path = kwargs['output_path']
    nProcess = kwargs['nProcess']
    nRun = kwargs['nRun']
    n_iter = kwargs['n_iter']

    arr_mean_energy_wo_decay = np.nan * np.ones(shape=n_iter)
    arr_mean_accuracy_wo_decay = np.nan * np.ones(shape=n_iter)

    arr_mean_energy_decay = np.nan * np.ones(shape=n_iter)
    arr_mean_accuracy_decay = np.nan * np.ones(shape=n_iter)

    # Benchmark
    benchmark_forgetting_all = Forgetting(nRun * nProcess, output_path + Constants.BENCHMARK_FORGETTING)

    # Catastrophic forgetting 1
    cat_forgetting_1_all = Forgetting(nRun * nProcess, output_path + Constants.CAT_FORGETTING_1)

    # Catastrophic forgetting 2
    cat_forgetting_2_all = Forgetting(nRun * nProcess, output_path + Constants.CAT_FORGETTING_2)

    # Active forgetting 1
    active_forgetting_1_all = Forgetting(nRun * nProcess, output_path + Constants.ACTIVE_FORGETTING_1)

    # Active forgetting 2
    active_forgetting_2_all = Forgetting(nRun * nProcess, output_path + Constants.ACTIVE_FORGETTING_2, n_iter)

    # Catastrophic forgetting 3
    cat_forgetting_3_all = Forgetting(nRun * nProcess, output_path + Constants.CAT_FORGETTING_3, n_iter)

    for iProcess in range(nProcess):
        start_index = iProcess * nRun
        end_index = start_index + nRun

        # 1. Benchmark
        arr_energy = np.loadtxt(output_path + Constants.BENCHMARK_FORGETTING +
                                Constants.ENERGY_FILE_PROC.format(str(iProcess)))
        arr_accuracy = np.loadtxt(output_path + Constants.BENCHMARK_FORGETTING +
                                  Constants.ACCURACY_FILE_PROC.format(str(iProcess)))
        benchmark_forgetting_all.energy[start_index:end_index] = arr_energy.reshape(-1, 1)
        benchmark_forgetting_all.accuracy[start_index:end_index] = arr_accuracy.reshape(-1, 1)

        # 2. Catastrophic forgetting - 1
        arr_energy = np.loadtxt(output_path + Constants.CAT_FORGETTING_1 +
                                Constants.ENERGY_FILE_PROC.format(str(iProcess)))
        arr_accuracy = np.loadtxt(output_path + Constants.CAT_FORGETTING_1 +
                                  Constants.ACCURACY_FILE_PROC.format(str(iProcess)))
        cat_forgetting_1_all.energy[start_index:end_index] = arr_energy.reshape(-1, 1)
        cat_forgetting_1_all.accuracy[start_index:end_index] = arr_accuracy.reshape(-1, 1)

        # 3. Catastrophic forgetting - 2
        arr_energy = np.loadtxt(output_path + Constants.CAT_FORGETTING_2 +
                                Constants.ENERGY_FILE_PROC.format(str(iProcess)))
        arr_accuracy = np.loadtxt(output_path + Constants.CAT_FORGETTING_2 +
                                  Constants.ACCURACY_FILE_PROC.format(str(iProcess)))
        cat_forgetting_2_all.energy[start_index:end_index] = arr_energy.reshape(-1, 1)
        cat_forgetting_2_all.accuracy[start_index:end_index] = arr_accuracy.reshape(-1, 1)

        # 4. Catastrophic forgetting - 3
        arr_energy = np.loadtxt(output_path + Constants.CAT_FORGETTING_3 +
                                Constants.ENERGY_FILE_PROC.format(str(iProcess)))
        arr_accuracy = np.loadtxt(output_path + Constants.CAT_FORGETTING_3 +
                                  Constants.ACCURACY_FILE_PROC.format(str(iProcess)))
        cat_forgetting_3_all.energy[start_index:end_index, :] = arr_energy
        cat_forgetting_3_all.accuracy[start_index:end_index, :] = arr_accuracy

        # 5. Active forgetting - 1
        arr_energy = np.loadtxt(output_path + Constants.ACTIVE_FORGETTING_1 +
                                Constants.ENERGY_FILE_PROC.format(str(iProcess)))
        arr_accuracy = np.loadtxt(output_path + Constants.ACTIVE_FORGETTING_1 +
                                  Constants.ACCURACY_FILE_PROC.format(str(iProcess)))
        active_forgetting_1_all.energy[start_index:end_index] = arr_energy.reshape(-1, 1)
        active_forgetting_1_all.accuracy[start_index:end_index] = arr_accuracy.reshape(-1, 1)

        # 6. Active forgetting - 2
        arr_energy = np.loadtxt(output_path + Constants.ACTIVE_FORGETTING_2 +
                                Constants.ENERGY_FILE_PROC.format(str(iProcess)))
        arr_accuracy = np.loadtxt(output_path + Constants.ACTIVE_FORGETTING_2 +
                                  Constants.ACCURACY_FILE_PROC.format(str(iProcess)))
        active_forgetting_2_all.energy[start_index:end_index, :] = arr_energy
        active_forgetting_2_all.accuracy[start_index:end_index, :] = arr_accuracy

    for i_iter in range(n_iter):
        arr_mean_accuracy_wo_decay[i_iter] = np.mean(cat_forgetting_3_all.accuracy[:, i_iter])
        arr_mean_energy_wo_decay[i_iter] = np.mean(cat_forgetting_3_all.energy[:, i_iter])

        arr_mean_accuracy_decay[i_iter] = np.mean(active_forgetting_2_all.accuracy[:, i_iter])
        arr_mean_energy_decay[i_iter] = np.mean(active_forgetting_2_all.energy[:, i_iter])

    # Save the mean values from all the runs to files.
    np.savetxt(output_path + Constants.BENCHMARK_FORGETTING + Constants.ENERGY_FILE,
               np.array([np.mean(benchmark_forgetting_all.energy)]))
    np.savetxt(output_path + Constants.BENCHMARK_FORGETTING + Constants.ACCURACY_FILE,
               np.array([np.mean(benchmark_forgetting_all.accuracy)]))

    np.savetxt(output_path + Constants.CAT_FORGETTING_1 + Constants.ENERGY_FILE,
               np.array([np.mean(cat_forgetting_1_all.energy)]))
    np.savetxt(output_path + Constants.CAT_FORGETTING_1 + Constants.ACCURACY_FILE,
               np.array([np.mean(cat_forgetting_1_all.accuracy)]))

    np.savetxt(output_path + Constants.CAT_FORGETTING_2 + Constants.ENERGY_FILE,
               np.array([np.mean(cat_forgetting_2_all.energy)]))
    np.savetxt(output_path + Constants.CAT_FORGETTING_2 + Constants.ACCURACY_FILE,
               np.array([np.mean(cat_forgetting_2_all.accuracy)]))

    np.savetxt(output_path + Constants.CAT_FORGETTING_3 + Constants.ENERGY_FILE, arr_mean_energy_wo_decay)
    np.savetxt(output_path + Constants.CAT_FORGETTING_3 + Constants.ACCURACY_FILE, arr_mean_accuracy_wo_decay)

    np.savetxt(output_path + Constants.ACTIVE_FORGETTING_1 + Constants.ENERGY_FILE,
               np.array([np.mean(active_forgetting_1_all.energy)]))
    np.savetxt(output_path + Constants.ACTIVE_FORGETTING_1 + Constants.ACCURACY_FILE,
               np.array([np.mean(active_forgetting_1_all.accuracy)]))

    np.savetxt(output_path + Constants.ACTIVE_FORGETTING_2 + Constants.ENERGY_FILE, arr_mean_energy_decay)
    np.savetxt(output_path + Constants.ACTIVE_FORGETTING_2 + Constants.ACCURACY_FILE, arr_mean_accuracy_decay)


def combine_perceptron_accuracy_results(**kwargs):
    decay_rates_lLTP = kwargs['decay_rates_lLTP']
    output_path = kwargs['output_path']
    nProcess = kwargs['nProcess']

    # Combined mean and std values
    arr_combined_mean_accuracy = np.nan * np.ones(shape=len(decay_rates_lLTP))
    arr_combined_mean_error = np.nan * np.ones(shape=len(decay_rates_lLTP))
    arr_combined_mean_energy = np.nan * np.ones(shape=(len(decay_rates_lLTP)))
    arr_combined_mean_epoch = np.nan * np.ones(shape=(len(decay_rates_lLTP)))

    arr_combined_std_accuracy = np.nan * np.ones(shape=len(decay_rates_lLTP))
    arr_combined_std_error = np.nan * np.ones(shape=len(decay_rates_lLTP))
    arr_combined_std_energy = np.nan * np.ones(shape=len(decay_rates_lLTP))
    arr_combined_std_epoch = np.nan * np.ones(shape=len(decay_rates_lLTP))

    # Arrays to store mean values from all processes
    arr_mean_accuracy = np.nan * np.ones(shape=(nProcess, len(decay_rates_lLTP)))
    arr_mean_error = np.nan * np.ones(shape=(nProcess, len(decay_rates_lLTP)))

    # Arrays to store all the run values from all the processes
    arr_all_accuracy = None
    arr_all_error = None
    arr_all_energy = None
    arr_all_epoch = None

    axis = 1 if len(decay_rates_lLTP) > 1 else 0

    for iProcess in range(nProcess):
        arr_mean_accuracy[iProcess] = np.loadtxt(output_path + Constants.ACCURACY_FILE_PROC.format(str(iProcess)))
        arr_mean_error[iProcess] = np.loadtxt(output_path + Constants.ERROR_FILE_PROC.format(str(iProcess)))

        # Read values from all the runs
        new_arr_accuracy = np.loadtxt(output_path + Constants.ACCURACY_FILE_PROC_ALL.format(str(iProcess)))
        new_arr_error = np.loadtxt(output_path + Constants.ERROR_FILE_PROC_ALL.format(str(iProcess)))
        new_arr_energy = np.loadtxt(output_path + Constants.ENERGY_FILE_PROC_ALL.format(str(iProcess)))
        new_arr_epoch = np.loadtxt(output_path + Constants.EPOCH_FILE_PROC_ALL.format(str(iProcess)))
        if arr_all_accuracy is None:
            arr_all_accuracy = new_arr_accuracy
        else:
            arr_all_accuracy = np.append(arr_all_accuracy, new_arr_accuracy, axis=axis)

        if arr_all_error is None:
            arr_all_error = new_arr_error
        else:
            arr_all_error = np.append(arr_all_error, new_arr_error, axis=axis)

        if arr_all_energy is None:
            arr_all_energy = new_arr_energy
        else:
            arr_all_energy = np.append(arr_all_energy, new_arr_energy, axis=axis)

        if arr_all_epoch is None:
            arr_all_epoch = new_arr_epoch
        else:
            arr_all_epoch = np.append(arr_all_epoch, new_arr_epoch, axis=axis)

    if len(decay_rates_lLTP) == 1:
        arr_combined_mean_accuracy[0] = np.mean(arr_all_accuracy)
        arr_combined_mean_error[0] = np.mean(arr_all_error)

        arr_combined_mean_energy[0] = np.mean(arr_all_energy)
        arr_combined_mean_epoch[0] = np.mean(arr_all_epoch)

        arr_combined_std_accuracy[0] = np.std(arr_all_accuracy)
        arr_combined_std_error[0] = np.std(arr_all_error)
        arr_combined_std_energy[0] = np.std(arr_all_energy)
        arr_combined_std_epoch[0] = np.std(arr_all_epoch)
    else:
        # Calculate the overall mean
        for i in range(len(decay_rates_lLTP)):
            arr_combined_mean_accuracy[i] = np.mean(arr_mean_accuracy[:, i])
            arr_combined_mean_error[i] = np.mean(arr_mean_error[:, i])

            arr_combined_mean_energy[i] = np.mean(arr_all_energy[i, :])
            arr_combined_mean_epoch[i] = np.mean(arr_all_epoch[i, :])

            arr_combined_std_accuracy[i] = np.std(arr_all_accuracy[i, :])
            arr_combined_std_error[i] = np.std(arr_all_error[i, :])
            arr_combined_std_energy[i] = np.std(arr_all_energy[i, :])
            arr_combined_std_epoch[i] = np.std(arr_all_epoch[i, :])

    np.savetxt(output_path + Constants.ACCURACY_FILE, arr_combined_mean_accuracy)
    np.savetxt(output_path + Constants.ERROR_FILE, arr_combined_mean_error)
    np.savetxt(output_path + Constants.ENERGY_FILE, arr_combined_mean_energy)
    np.savetxt(output_path + Constants.EPOCH_FILE, arr_combined_mean_epoch)

    np.savetxt(output_path + Constants.STD_ACCURACY_FILE, arr_combined_std_accuracy)
    np.savetxt(output_path + Constants.STD_ERROR_FILE, arr_combined_std_error)
    np.savetxt(output_path + Constants.STD_ENERGY_FILE, arr_combined_std_energy)
    np.savetxt(output_path + Constants.STD_EPOCH_FILE, arr_combined_std_epoch)


def combine_results(output_path_1, output_path_2, keys):
    arr = np.loadtxt(output_path_1)
    arr = np.concatenate((arr, np.loadtxt(output_path_2)))
    dict_combined = dict(zip(keys, arr))
    dict_combined = OrderedDict(sorted(dict_combined.items()))

    return dict_combined


def combine_perceptron_decay_results(output_path_1, output_path_2, res_output_path, is_accuracy=True):
    Path(res_output_path).mkdir(parents=True, exist_ok=True)
    decay_rates_lLTP = np.loadtxt(output_path_1 + Constants.DECAY_RATES_FILE)
    decay_rates_lLTP = np.concatenate((decay_rates_lLTP, np.loadtxt(output_path_2 + Constants.DECAY_RATES_FILE)))

    if is_accuracy:
        # Accuracy
        dict_accuracy = combine_results(output_path_1 + Constants.ACCURACY_FILE,
                                        output_path_2 + Constants.ACCURACY_FILE, decay_rates_lLTP)
        np.savetxt(res_output_path + Constants.DECAY_RATES_FILE, [*dict_accuracy.keys()])
        np.savetxt(res_output_path + Constants.ACCURACY_FILE, [*dict_accuracy.values()])

        dict_std_accuracy = combine_results(output_path_1 + Constants.STD_ACCURACY_FILE,
                                            output_path_2 + Constants.STD_ACCURACY_FILE, decay_rates_lLTP)
        np.savetxt(res_output_path + Constants.STD_ACCURACY_FILE, [*dict_std_accuracy.values()])
    else:
        # Max number of patterns
        dict_patterns = combine_results(output_path_1 + Constants.PATTERNS_FILE,
                                        output_path_2 + Constants.PATTERNS_FILE, decay_rates_lLTP)
        np.savetxt(res_output_path + Constants.DECAY_RATES_FILE, [*dict_patterns.keys()])
        np.savetxt(res_output_path + Constants.PATTERNS_FILE, [*dict_patterns.values()])

        dict_std_patterns = combine_results(output_path_1 + Constants.STD_PATTERNS_FILE,
                                            output_path_2 + Constants.STD_PATTERNS_FILE, decay_rates_lLTP)
        np.savetxt(res_output_path + Constants.STD_PATTERNS_FILE, [*dict_std_patterns.values()])

    # Energy
    dict_energy = combine_results(output_path_1 + Constants.ENERGY_FILE, output_path_2 + Constants.ENERGY_FILE,
                                  decay_rates_lLTP)
    np.savetxt(res_output_path + Constants.ENERGY_FILE, [*dict_energy.values()])

    dict_std_energy = combine_results(output_path_1 + Constants.STD_ENERGY_FILE,
                                      output_path_2 + Constants.STD_ENERGY_FILE, decay_rates_lLTP)
    np.savetxt(res_output_path + Constants.STD_ENERGY_FILE, [*dict_std_energy.values()])

    # Error
    if is_accuracy:
        dict_error = combine_results(output_path_1 + Constants.ERROR_FILE, output_path_2 + Constants.ERROR_FILE,
                                     decay_rates_lLTP)
        np.savetxt(res_output_path + Constants.ERROR_FILE, [*dict_error.values()])

        dict_std_error = combine_results(output_path_1 + Constants.STD_ERROR_FILE,
                                         output_path_2 + Constants.STD_ERROR_FILE, decay_rates_lLTP)
        np.savetxt(res_output_path + Constants.STD_ERROR_FILE, [*dict_std_error.values()])

    # Epoch
    dict_epoch = combine_results(output_path_1 + Constants.EPOCH_FILE, output_path_2 + Constants.EPOCH_FILE,
                                 decay_rates_lLTP)
    np.savetxt(res_output_path + Constants.EPOCH_FILE, [*dict_epoch.values()])

    dict_std_epoch = combine_results(output_path_1 + Constants.STD_EPOCH_FILE, output_path_2 + Constants.STD_EPOCH_FILE,
                                     decay_rates_lLTP)
    np.savetxt(res_output_path + Constants.STD_EPOCH_FILE, [*dict_std_epoch.values()])


def perm_decay_wrapper_process(**kwargs):
    nProcess = kwargs['nProcess']
    args = [iProcess for iProcess in range(nProcess)]
    pool = mp.Pool(nProcess)
    pool.map(partial(perm_decay_rates, **kwargs), args)
    pool.close()

    # Combine the results from all the processes
    combine_perm_decay_results(**kwargs)


def perceptron_accuracy_wrapper_process(**kwargs):
    nProcess = kwargs['nProcess']
    args = [iProcess for iProcess in range(nProcess)]
    pool = mp.Pool(nProcess)
    pool.map(partial(perceptron_accuracy, **kwargs), args)
    pool.close()

    # Combine the results from all the processes
    combine_perceptron_accuracy_results(**kwargs)


def perceptron_forgetting_wrapper_process(**kwargs):
    nProcess = kwargs['nProcess']
    args = [iProcess for iProcess in range(nProcess)]
    pool = mp.Pool(nProcess)
    pool.map(partial(perceptron_forgetting, **kwargs), args)
    pool.close()

    # Combine the results from all the processes
    combine_perceptron_forgetting_results(**kwargs)


def perm_decay_wrapper(nDimension, iPattern_init, step_size, decay_rates_lLTP, dir_name):
    nPattern = 2 * nDimension  # max capacity of the perceptron
    # decay_rates_lLTP = np.logspace(-6, -4, 30)
    # decay_rates_lLTP = np.logspace(-4, -2, 30)
    # decay_rates_lLTP = np.logspace(-2, -1, 10)
    output_path = Constants.PERM_DECAY_PATH + '/dim_' + str(nDimension) + dir_name
    perm_decay_wrapper_process(nDimension=nDimension, nPattern=nPattern, iPattern_init=iPattern_init,
                               step_size=step_size,
                               decay_rates_lLTP=decay_rates_lLTP, output_path=output_path, nRun=10, nProcess=5)
    np.savetxt(output_path + Constants.DECAY_RATES_FILE, decay_rates_lLTP)


def perceptron_accuracy_wrapper(nDimension, nPattern, decay_rates_lLTP, dir_name, epoch_updates=False):
    # decay_rates_lLTP = np.logspace(-6, -4, 30)
    # decay_rates_lLTP = np.array([1e-6, 1e-5, 1e-4])
    # decay_rates_lLTP = np.logspace(-4, -2, 30)
    # decay_rates_lLTP = [0.0]
    # decay_rates_lLTP = np.logspace(-8, -6, 30)
    output_path = Constants.PERM_DECAY_ACCURACY_PATH + dir_name  # '/low_decay'
    perceptron_accuracy_wrapper_process(nDimension=nDimension, nPattern=nPattern, decay_rates_lLTP=decay_rates_lLTP,
                                        output_path=output_path, window_size=25, nRun=10, nProcess=5,
                                        epoch_updates=epoch_updates)
    np.savetxt(output_path + Constants.DECAY_RATES_FILE, decay_rates_lLTP)


def perceptron_forgetting_wrapper(nDimension, nPattern, dir_name, new_patterns=100, n_iter=10, decay_rate=1e-6):
    output_path = Constants.PERM_DECAY_FORGETTING_PATH + dir_name
    perceptron_forgetting_wrapper_process(nDimension=nDimension, nPattern=nPattern, output_path=output_path,
                                          window_size=25, nRun=10, nProcess=5, new_patterns_count=new_patterns,
                                          n_iter=n_iter, decay_rate=decay_rate)
