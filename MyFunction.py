#!/usr/bin/env python
import math
import numpy as np


class Perceptron():

    def __init__(self, pattern, pattern_answer, weight_initial, size_buffer,
                 learning_rate, energy_scale_lLTP, energy_detail=None):
        self.pattern = pattern
        self.pattern_answer = pattern_answer
        self.weight_initial = weight_initial
        self.size_buffer = size_buffer
        self.learning_rate = learning_rate
        self.energy_scale_lLTP = energy_scale_lLTP
        self.energy_detail = energy_detail
        self.nDim = len(weight_initial) - 1
        self.nPattern = len(pattern_answer)
        self.synapse_threshold = 0.05  # if dW of a synapse > thres, W of that syn updated
        self.energy_scale_maintenance = 0.
        self.decay_eLTP = 0.
        self.decay_lLTP = 0.

        self.arr_weight = np.zeros(self.nDim + 1)
        self.arr_deltaW = np.zeros(self.nDim + 1)
        self.arr_deltaW_previous = np.zeros(self.nDim + 1)
        self.arr_weight_buffer = -999 * np.ones(shape=(size_buffer, self.nDim + 1))
        self.arr_energy_buffer = 999 * np.ones(size_buffer)
        self.arr_energy_eLTP_buffer = 999 * np.ones(size_buffer)
        self.arr_energy_lLTP_buffer = 999 * np.ones(size_buffer)
        self.arr_count_error_buffer = 999 * np.ones(size_buffer)
        self.count_error = 999

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
        self.arr_weight = np.copy(self.weight_initial)
        self.arr_deltaW = np.zeros(self.nDim + 1)
        self.arr_weight_buffer = -999 * np.ones(shape=(self.size_buffer, self.nDim + 1))
        self.arr_energy_buffer = 999 * np.ones(self.size_buffer)
        self.arr_energy_eLTP_buffer = 999 * np.ones(self.size_buffer)
        self.arr_energy_lLTP_buffer = 999 * np.ones(self.size_buffer)
        self.arr_count_error_buffer = 999 * np.ones(self.size_buffer)
        self.count_error = 999

    def CalculateOutput(self, iPattern):
        # adding maintenance cost into var_energy and var_energy_eLTP
        self.var_energy += self.energy_scale_maintenance * np.sum(np.fabs(self.arr_weight[np.arange(0, self.nDim)]))
        # self.var_energy_eLTP += self.energy_scale_maintenance * np.sum(
        #     np.fabs(self.arr_deltaW[np.arange(0, self.nDim)]))
        # self.arr_deltaW[np.arange(0, self.nDim)] = self.arr_deltaW[np.arange(0, self.nDim)] * np.exp(-self.decay_eLTP)
        self.arr_weight[np.arange(0, self.nDim)] = self.arr_weight[np.arange(0, self.nDim)] * np.exp(-self.decay_lLTP)
        input = np.dot(self.arr_weight, self.pattern[iPattern])
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

    def BreakLoop(self):
        tmp = self.var_epoch % self.size_buffer
        if (self.count_error >= self.arr_count_error_buffer[tmp]):
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
        # self.arr_weight += self.arr_deltaW
        self.var_energy += self.energy_scale_lLTP * np.sum(
            np.fabs(self.arr_weight[np.arange(0, self.nDim)]))  # assume changing bias costs no energy
        # self.var_energy_lLTP += self.energy_scale_lLTP * np.sum(
        #     np.fabs(self.arr_deltaW[np.arange(0, self.nDim)]))  # assume changing bias costs no energy

    def Output(self):
        if (self.energy_detail is None):
            return self.var_energy, self.var_error, self.var_epoch
        else:
            return self.var_energy, self.var_energy_eLTP, self.var_energy_lLTP, self.var_error, self.var_epoch

    ####################################################
    #                                                  #
    #                    Algorithms                    #
    #                                                  #
    ####################################################

    # standard perceptron
    def AlgoStandard(self):
        self.Initialise()
        while (self.count_error != 0):
            self.count_error = 0.;
            if_spike = 0
            self.arr_deltaW = np.zeros(self.nDim + 1)
            for iPattern in range(0, self.nPattern):
                difference = self.CalculateOutput(iPattern)
                if (difference != 0):
                    self.arr_deltaW = self.learning_rate * difference * self.pattern[iPattern]
                    self.SaveChange()
                    self.count_error += 1.
            if (self.BreakLoop()):
                break
        self.Finalise()
        return self.Output()

    # only change individual synapse w when deltaW is large
    def AlgoSynapse(self, synapse_threshold=None):
        if (synapse_threshold is None):
            print("Warning: synapse threshold is not set manually!")
            synapse_threshold = self.synapse_threshold
        self.Initialise()
        while (self.count_error != 0):
            self.count_error = 0.;
            if_spike = 0
            for iPattern in range(0, self.nPattern):
                difference = self.CalculateOutput(iPattern)
                if (difference != 0):
                    self.arr_deltaW += self.learning_rate * difference * self.pattern[iPattern]
                    self.count_error += 1.
                arr_syn_update = np.where(np.fabs(self.arr_deltaW) > synapse_threshold)[0]
                for iArrSyn in range(0, len(arr_syn_update)):
                    nSynapse = arr_syn_update[iArrSyn]
                    self.arr_weight[nSynapse] += self.arr_deltaW[nSynapse]
                    if (nSynapse != self.nDim):
                        self.var_energy += self.energy_scale_lLTP * np.fabs(
                            self.arr_deltaW[nSynapse])  # assume changing bias costs no energy
                        self.var_energy_lLTP += self.energy_scale_lLTP * np.fabs(
                            self.arr_deltaW[nSynapse])  # assume changing bias costs no energy
                    self.arr_deltaW[nSynapse] = 0.
            if (self.BreakLoop()):
                break
        self.Finalise()
        return self.Output()

    # update w (all synapses) when an individual synapse reaches the threshold
    def AlgoSynapseAll(self, synapse_threshold=None):
        if synapse_threshold is None:
            print("Warning: synapse threshold is not set manually!")
            synapse_threshold = self.synapse_threshold
        self.Initialise()
        while self.count_error != 0:
            self.count_error = 0.;
            for iPattern in range(0, self.nPattern):
                difference = self.CalculateOutput(iPattern)
                if difference != 0:
                    self.arr_weight += self.learning_rate * difference * self.pattern[iPattern]
                    self.count_error += 1
                # arr_syn_update = np.where(np.fabs(self.arr_deltaW) > synapse_threshold)[0]
                # if len(arr_syn_update) > 0:
                #     self.SaveChange()
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
        if (is_pattern_integer):
            if (is_plus_minus_one):  # set patterns to either +1 or -1
                pattern[iPattern] = -1 + 2 * np.around(np.random.uniform(0, 1, nDimension + 1))
            else:  # set patterns to either 1 or 0
                pattern[iPattern] = np.where(np.random.uniform(-1, 1., nDimension + 1) < 0., 0, 1)
        else:  # set patterns to random values
            pattern[iPattern] = np.around(np.random.uniform(-1, 1, nDimension + 1), 3)
        pattern[iPattern][nDimension] = 1.
        pattern_answer[iPattern] = int(np.random.uniform(-1, 1) > 0.)
    return pattern, pattern_answer
