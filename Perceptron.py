#!/usr/bin/env python
import numpy as np
from collections import deque
import itertools
import Code.log_helper as log_helper

logger = log_helper.get_logger('Perceptron')


class Perceptron:

    def __init__(self, pattern, pattern_answer, weight_initial, size_buffer,
                 learning_rate, energy_scale_lLTP, energy_detail=None, use_accuracy=True):
        self.pattern = np.copy(pattern)
        self.pattern_answer = np.copy(pattern_answer)
        self.weight_initial = weight_initial
        self.size_buffer = size_buffer
        self.epoch_window = 200
        self.learning_rate = learning_rate
        self.energy_scale_lLTP = energy_scale_lLTP
        self.energy_detail = energy_detail
        self.nDim = len(weight_initial) - 1
        self.nPattern = len(pattern_answer)
        self.synapse_threshold = 0.05  # if dW of a synapse > thres, W of that syn updated
        self.energy_scale_maintenance = 0.
        self.decay_eLTP = 0.
        self.decay_lLTP = 0.
        self.use_accuracy = use_accuracy
        self.initialize_weights = True

        self.arr_weight = np.zeros(self.nDim + 1)
        self.arr_deltaW = np.zeros(self.nDim + 1)
        self.arr_deltaW_previous = np.zeros(self.nDim + 1)
        self.arr_weight_buffer = -999 * np.ones(shape=(size_buffer, self.nDim + 1))
        self.arr_energy_buffer = 999 * np.ones(size_buffer)
        self.arr_energy_eLTP_buffer = 999 * np.ones(size_buffer)
        self.arr_energy_lLTP_buffer = 999 * np.ones(size_buffer)
        self.arr_count_error_buffer = 999 * np.ones(size_buffer)
        self.accuracy_window_buffer = deque((self.epoch_window * 2) * [0.], maxlen=self.epoch_window * 2)
        self.error_window_buffer = deque((self.epoch_window * 2) * [0.], maxlen=self.epoch_window * 2)

        self.count_updates = 999
        self.count_updates_np = 999
        self.arr_epoch_updates = []
        self.arr_epoch_updates_np = []
        self.arr_energy_updates = []
        self.count_error = 999
        self.count_correctly_classified = 999
        self.accuracy = 999

        self.var_energy = 0.
        self.var_energy_eLTP = 0.
        self.var_energy_lLTP = 0.
        self.var_error = 0.
        self.var_epoch = 0

    def Initialise(self):
        self.var_energy = 0.
        self.var_energy_eLTP = 0.
        self.var_energy_lLTP = 0.
        self.var_error = 0.
        self.var_epoch = 0

        if self.initialize_weights:
            self.arr_weight = np.copy(self.weight_initial)

        self.arr_deltaW = np.zeros(self.nDim + 1)
        self.arr_weight_buffer = -999 * np.ones(shape=(self.size_buffer, self.nDim + 1))
        self.arr_energy_buffer = 999 * np.ones(self.size_buffer)
        self.arr_energy_eLTP_buffer = 999 * np.ones(self.size_buffer)
        self.arr_energy_lLTP_buffer = 999 * np.ones(self.size_buffer)
        self.arr_count_error_buffer = 999 * np.ones(self.size_buffer)
        self.accuracy_window_buffer = deque((self.epoch_window * 2) * [0.], maxlen=self.epoch_window * 2)
        self.error_window_buffer = deque((self.epoch_window * 2) * [0.], maxlen=self.epoch_window * 2)
        self.count_error = 999
        self.count_correctly_classified = 999
        self.accuracy = 999

        self.count_updates = 999
        self.count_updates_np = 999
        self.arr_epoch_updates = []
        self.arr_epoch_updates_np = []
        self.arr_energy_updates = []

    def CalculateOutput(self, iPattern):
        # adding maintenance cost into var_energy and var_energy_eLTP
        self.var_energy += self.energy_scale_maintenance * np.sum(np.fabs(self.arr_deltaW[np.arange(0, self.nDim)]))
        self.var_energy_eLTP += self.energy_scale_maintenance * np.sum(
            np.fabs(self.arr_deltaW[np.arange(0, self.nDim)]))
        self.arr_deltaW[np.arange(0, self.nDim)] = self.arr_deltaW[np.arange(0, self.nDim)] * np.exp(-self.decay_eLTP)
        self.arr_weight[np.arange(0, self.nDim)] = self.arr_weight[np.arange(0, self.nDim)] * np.exp(-self.decay_lLTP)
        input = np.dot(self.arr_weight + self.arr_deltaW, self.pattern[iPattern])
        if_spike = int(input > 0.)
        difference = self.pattern_answer[iPattern] - if_spike
        return difference

    def SaveChange(self):
        self.arr_weight += self.arr_deltaW
        self.var_energy += self.energy_scale_lLTP * np.sum(
            np.fabs(self.arr_deltaW[np.arange(0, self.nDim)]))  # assume changing bias costs no energy
        self.var_energy_lLTP += self.energy_scale_lLTP * np.sum(
            np.fabs(self.arr_deltaW[np.arange(0, self.nDim)]))  # assume changing bias costs no energy
        self.arr_deltaW = np.zeros(self.nDim + 1)
        self.count_updates += 1

    def BreakLoop(self):
        if self.use_accuracy:
            # Use the value of accuracy to quit training
            self.accuracy = self.count_correctly_classified / self.nPattern
            self.accuracy_window_buffer.append(self.accuracy)
            self.error_window_buffer.append(self.count_error)

            self.arr_epoch_updates.append(self.count_updates)
            self.arr_epoch_updates_np.append(self.count_updates_np)
            self.arr_energy_updates.append(self.var_energy[0])
            self.var_epoch += 1

            if self.var_epoch >= (2 * self.epoch_window):
                mean_accuracy = np.mean(
                    np.fromiter(itertools.islice(self.accuracy_window_buffer, self.epoch_window, None), float))
                mean_accuracy_prev = np.mean(
                    np.fromiter(itertools.islice(self.accuracy_window_buffer, self.epoch_window), float))

                if mean_accuracy_prev > mean_accuracy:
                    logger.info(f'No improvement in mean accuracy. Quitting! Mean accuracy: {mean_accuracy} Mean '
                                f'previous accuracy: {mean_accuracy_prev}')
                    return True

            return False
        else:
            # Quit the training when the training error doesn't improve
            tmp = self.var_epoch % self.size_buffer
            if self.count_error >= self.arr_count_error_buffer[tmp]:
                self.var_epoch -= self.size_buffer
                self.arr_weight = self.arr_weight_buffer[tmp]
                self.var_energy = self.arr_energy_buffer[tmp]
                self.var_energy_eLTP = self.arr_energy_eLTP_buffer[tmp]
                self.var_energy_lLTP = self.arr_energy_lLTP_buffer[tmp]
                return True  # when error rate doesn't improve anymore
            else:
                self.arr_count_error_buffer[tmp] = self.count_error
                self.arr_weight_buffer[tmp] = self.arr_weight
                self.arr_energy_buffer[tmp] = self.var_energy
                self.arr_energy_eLTP_buffer[tmp] = self.var_energy_eLTP
                self.arr_energy_lLTP_buffer[tmp] = self.var_energy_lLTP
                self.var_epoch += 1

                return False

    def Finalise(self):
        self.var_error += self.count_error / self.nPattern

        # consolidate all e-LTP weights to l-LTP weights at the end
        self.arr_weight += self.arr_deltaW
        self.var_energy += self.energy_scale_lLTP * np.sum(
            np.fabs(self.arr_deltaW[np.arange(0, self.nDim)]))  # assume changing bias costs no energy
        self.var_energy_lLTP += self.energy_scale_lLTP * np.sum(
            np.fabs(self.arr_deltaW[np.arange(0, self.nDim)]))  # assume changing bias costs no energy

    def Output(self):
        if self.energy_detail is None:
            return self.var_energy, self.var_error, self.var_epoch
        else:
            if self.use_accuracy:
                return self.var_energy, self.var_energy_eLTP, self.var_energy_lLTP, self.var_error, self.var_epoch, \
                       self.accuracy, self.arr_epoch_updates, self.arr_energy_updates
            else:
                return self.var_energy, self.var_energy_eLTP, self.var_energy_lLTP, self.var_error, self.var_epoch

    ####################################################
    #                                                  #
    #                    Algorithms                    #
    #                                                  #
    ####################################################

    # standard perceptron
    def AlgoStandard(self, start_index=None, end_index=None):
        if start_index is None:
            start_index = 0
        if end_index is None:
            end_index = self.nPattern - 1
        self.Initialise()
        while self.count_error != 0:
            self.count_error = 0.
            self.count_correctly_classified = 0.
            self.count_updates = 0.
            self.count_updates_np = 0.
            self.arr_deltaW = np.zeros(self.nDim + 1)
            for iPattern in range(0, self.nPattern):
                difference = self.CalculateOutput(iPattern)
                if difference != 0:
                    self.arr_deltaW = self.learning_rate * difference * self.pattern[iPattern]
                    self.SaveChange()
                    self.count_error += 1.
                    if start_index <= iPattern < end_index:
                        self.count_updates_np += 1
                else:
                    self.count_correctly_classified += 1
            if self.BreakLoop():
                break
        self.Finalise()
        return self.Output()

    # only change individual synapse w when deltaW is large
    def AlgoSynapse(self, synapse_threshold=None):
        if synapse_threshold is None:
            logger.warn("Warning: synapse threshold is not set manually!")
            synapse_threshold = self.synapse_threshold
        self.Initialise()
        while self.count_error != 0:
            self.count_error = 0.
            for iPattern in range(0, self.nPattern):
                difference = self.CalculateOutput(iPattern)
                if difference != 0:
                    self.arr_deltaW += self.learning_rate * difference * self.pattern[iPattern]
                    self.count_error += 1.
                arr_syn_update = np.where(np.fabs(self.arr_deltaW) > synapse_threshold)[0]
                for iArrSyn in range(0, len(arr_syn_update)):
                    nSynapse = arr_syn_update[iArrSyn]
                    self.arr_weight[nSynapse] += self.arr_deltaW[nSynapse]
                    if nSynapse != self.nDim:
                        self.var_energy += self.energy_scale_lLTP * np.fabs(
                            self.arr_deltaW[nSynapse])  # assume changing bias costs no energy
                        self.var_energy_lLTP += self.energy_scale_lLTP * np.fabs(
                            self.arr_deltaW[nSynapse])  # assume changing bias costs no energy
                    self.arr_deltaW[nSynapse] = 0.
            if self.BreakLoop():
                break
        self.Finalise()
        return self.Output()

    # update w (all synapses) when an individual synapse reaches the threshold
    def AlgoSynapseAll(self, synapse_threshold=None):
        if synapse_threshold is None:
            logger.warn("Warning: synapse threshold is not set manually!")
            synapse_threshold = self.synapse_threshold
        self.Initialise()
        while self.count_error != 0:
            self.count_error = 0.
            for iPattern in range(0, self.nPattern):
                difference = self.CalculateOutput(iPattern)
                if difference != 0:
                    self.arr_deltaW += self.learning_rate * difference * self.pattern[iPattern]
                    self.count_error += 1
                arr_syn_update = np.where(np.fabs(self.arr_deltaW) > synapse_threshold)[0]
                if len(arr_syn_update) > 0:
                    self.SaveChange()
            if self.BreakLoop():
                break
        self.Finalise()
        return self.Output()


####################################################
#                                                  #
#                Pattern generation                #
#                                                  #
####################################################
def MakePattern(nPattern, nDimension, is_pattern_integer=False, is_plus_minus_one=False):
    pattern = np.ones(shape=(nPattern, nDimension + 1))  # +1 is from bias
    pattern_answer = -999 * np.ones(nPattern)
    for iPattern in range(0, nPattern):
        if is_pattern_integer:
            if is_plus_minus_one:  # set patterns to either +1 or -1
                pattern[iPattern] = -1 + 2 * np.around(np.random.uniform(0, 1, nDimension + 1))
            else:  # set patterns to either 1 or 0
                pattern[iPattern] = np.where(np.random.uniform(-1, 1., nDimension + 1) < 0., 0, 1)
        else:  # set patterns to random values
            pattern[iPattern] = np.around(np.random.uniform(-1, 1, nDimension + 1), 3)
        pattern[iPattern][nDimension] = 1.
        pattern_answer[iPattern] = int(np.random.uniform(-1, 1) > 0.)
    return pattern, pattern_answer
