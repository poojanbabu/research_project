from pathlib import Path

HOME_DIR = str(Path.home())

############## TEXT OUTPUT ################

BASE_OUTPUT_PATH = HOME_DIR + '/Research_Project/Shared/Text'
PERM_DECAY_PATH = BASE_OUTPUT_PATH + '/Perm_decay'
PERM_DECAY_ACCURACY_PATH = PERM_DECAY_PATH + '/accuracy'
PERM_DECAY_FORGETTING_PATH = PERM_DECAY_PATH + '/forgetting'

# Files for different measures
DECAY_RATES_FILE = '/decay_rates.txt'
ENERGY_FILE = '/energy.txt'
ENERGY_WITHOUT_DECAY_FILE = '/energy_without_decay.txt'
ERROR_FILE = "/error.txt"
EPOCH_FILE = '/epoch.txt'
PATTERNS_FILE = "/patterns.txt"
ACCURACY_FILE = '/accuracy.txt'
ACCURACY_WITHOUT_DECAY_FILE = '/accuracy_without_decay.txt'

# Mean values from individual process
ENERGY_FILE_PROC = '/energy_{}.txt'
ENERGY_FILE_WITHOUT_DECAY_PROC = '/energy_without_decay_{}.txt'
ENERGY_ELTP_FILE_PROC = '/energy_eLTP_{}.txt'
ENERGY_LLTP_FILE_PROC = '/energy_lLTP_{}.txt'
ERROR_FILE_PROC = "/error_{}.txt"
EPOCH_FILE_PROC = '/epoch_{}.txt'
PATTERNS_FILE_PROC = "/patterns_{}.txt"
ACCURACY_FILE_PROC = '/accuracy_{}.txt'
ACCURACY_FILE_WITHOUT_DECAY_PROC = '/accuracy_without_decay_{}.txt'

# Values from all runs from individual process
ENERGY_FILE_PROC_ALL = '/energy_all_{}.txt'
ENERGY_ELTP_FILE_PROC_ALL = '/energy_eLTP_all_{}.txt'
ENERGY_LLTP_FILE_PROC_ALL = '/energy_lLTP_all_{}.txt'
ERROR_FILE_PROC_ALL = "/error_all_{}.txt"
EPOCH_FILE_PROC_ALL = '/epoch_all_{}.txt'
ACCURACY_FILE_PROC_ALL = '/accuracy_all_{}.txt'

EPOCH_UPDATES_ALL = '/epoch_updates.npy'
EPOCH_UPDATES_NP_ALL = '/epoch_updates_np.npy'
EPOCH_UPDATES_WITHOUT_DECAY_ALL = '/epoch_updates_without_decay.npy'
EPOCH_UPDATES_NP_WITHOUT_DECAY_ALL = '/epoch_updates_np_without_decay.npy'
ENERGY_UPDATES_ALL = '/energy_updates.npy'
ENERGY_UPDATES_WITHOUT_DECAY_ALL = '/energy_updates_without_decay.npy'

# STD files
STD_EPOCH_FILE_PROC = "/std_epoch_{}.txt"
STD_EPOCH_FILE = "/std_epoch.txt"
STD_ENERGY_FILE = "/std_energy.txt"
STD_PATTERNS_FILE = "/std_patterns.txt"
STD_ACCURACY_FILE = '/std_accuracy.txt'
STD_ERROR_FILE = '/std_error.txt'

# Forgetting directories
BENCHMARK_FORGETTING = '/benchmark'
CAT_FORGETTING_1 = '/cat_forgetting_1'
CAT_FORGETTING_2 = '/cat_forgetting_2'
CAT_FORGETTING_3 = '/cat_forgetting_3'
PASSIVE_FORGETTING_1 = '/active_forgetting_1'
PASSIVE_FORGETTING_2 = '/active_forgetting_2'

################ PLOT ###################
BASE_PLOT_PATH = HOME_DIR + '/Research_Project/Shared/Plot'
PERM_DECAY_PLOT_PATH = BASE_PLOT_PATH + '/Perm_decay'
PERM_DECAY_ACCURACY_PLOT_PATH = PERM_DECAY_PLOT_PATH + '/accuracy'
PERM_DECAY_FORGETTING_PLOT_PATH = PERM_DECAY_PLOT_PATH + '/forgetting'

ENERGY_PLOT = '/energy.png'
ENERGY_BAR_PLOT = '/energy_bar.png'
ENERGY_FORGETTING_BAR_PLOT = '/energy_forgetting_bar.png'
ENERGY_FORGETTING_LINE_PLOT = '/energy_forgetting_line.png'
ENERGY_ELTP_PLOT = '/energy_eLTP.png'
ENERGY_LLTP_PLOT = '/energy_lLTP.png'
ERROR_PLOT = '/error.png'
EPOCH_PLOT = '/epoch.png'
PATTERNS_PLOT = '/patterns.png'
PATTERNS_SLOPE_PLOT = '/slope.png'
ACCURACY_PLOT = '/accuracy.png'
EPOCH_UPDATES_PLOT = '/epoch_updates.png'
ENERGY_UPDATES_PLOT = '/energy_updates.png'
ENERGY_PER_PATTERN_PLOT = '/energy_per_pattern.png'
ENERGY_PER_SYNAPSE_PLOT = '/energy_per_synapse.png'
