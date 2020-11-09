#!/usr/bin/env python
import numpy as np
from pathlib import Path

import Code.MyConstants as Constants


class Forgetting:
    def __init__(self, nRun, output_path, n_subsets=1, decay_rate=0.0, n_iter=1, initialize_all=False):
        self.energy = np.zeros(shape=(nRun, n_iter))
        self.accuracy = np.zeros(shape=(nRun, n_iter))
        self.decay_lLTP = decay_rate
        self.n_subsets = n_subsets
        self.output_path = output_path
        self.initialize_all = initialize_all

        if self.decay_lLTP != 0.0:
            if self.n_subsets > 1:
                self.output_path = self.output_path + '/' + str(self.decay_lLTP) + '/' + str(self.n_subsets)
            else:
                self.output_path = self.output_path + '/' + str(self.decay_lLTP)
        elif self.n_subsets > 1:
            self.output_path = self.output_path + '/' + str(self.n_subsets)

        if initialize_all:
            self.energy_eLTP = np.zeros(shape=(nRun, n_iter))
            self.energy_lLTP = np.zeros(shape=(nRun, n_iter))
            self.error = np.zeros(shape=(nRun, n_iter))
            self.epoch = np.zeros(shape=(nRun, n_iter))
            self.epoch_updates = []
            self.energy_updates = []

    def save_proc_output(self, iProcess):
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        np.savetxt(self.output_path + Constants.ACCURACY_FILE_PROC.format(str(iProcess)), self.accuracy)
        np.savetxt(self.output_path + Constants.ENERGY_FILE_PROC.format(str(iProcess)), self.energy)

        if self.initialize_all:
            np.savetxt(self.output_path + Constants.ENERGY_ELTP_FILE_PROC.format(str(iProcess)), self.energy_eLTP)
            np.savetxt(self.output_path + Constants.ENERGY_LLTP_FILE_PROC.format(str(iProcess)), self.energy_lLTP)
            np.savetxt(self.output_path + Constants.ERROR_FILE_PROC.format(str(iProcess)), self.error)
            np.savetxt(self.output_path + Constants.EPOCH_FILE_PROC.format(str(iProcess)), self.epoch)
