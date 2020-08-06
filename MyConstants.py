############## TEXT OUTPUT ################
BASE_OUTPUT_PATH = '/home/pooja/Research_Project/Shared/Text'
PERM_DECAY_PATH = BASE_OUTPUT_PATH + '/Perm_decay'
PERM_DECAY_ACCURACY_PATH = PERM_DECAY_PATH + '/accuracy'

# Files for different measures
DECAY_RATES_FILE = '/decay_rates.txt'
ENERGY_FILE = '/energy.txt'
ERROR_FILE = "/error.txt"
EPOCH_FILE = '/epoch.txt'
PATTERNS_FILE = "/patterns.txt"
ACCURACY_FILE = '/accuracy.txt'

# Mean values from individual process
ENERGY_FILE_PROC = '/energy_{}.txt'
ENERGY_ELTP_FILE_PROC = '/energy_eLTP_{}.txt'
ENERGY_LLTP_FILE_PROC = '/energy_lLTP_{}.txt'
ERROR_FILE_PROC = "/error_{}.txt"
EPOCH_FILE_PROC = '/epoch_{}.txt'
PATTERNS_FILE_PROC = "/patterns_{}.txt"
ACCURACY_FILE_PROC = '/accuracy_{}.txt'

# Values from all runs from individual process
ENERGY_FILE_PROC_ALL = '/energy_all_{}.txt'
ENERGY_ELTP_FILE_PROC_ALL = '/energy_eLTP_all_{}.txt'
ENERGY_LLTP_FILE_PROC_ALL = '/energy_lLTP_all_{}.txt'
ERROR_FILE_PROC_ALL = "/error_all_{}.txt"
EPOCH_FILE_PROC_ALL = '/epoch_all_{}.txt'
ACCURACY_FILE_PROC_ALL = '/accuracy_all_{}.txt'

EPOCH_UPDATES_ALL = '/epoch_updates.npy'
ENERGY_UPDATES_ALL = '/energy_updates.npy'

# STD files
STD_EPOCH_FILE_PROC = "/std_epoch_{}.txt"
STD_EPOCH_FILE = "/std_epoch.txt"
STD_ENERGY_FILE = "/std_energy.txt"
STD_PATTERNS_FILE = "/std_patterns.txt"
STD_ACCURACY_FILE = '/std_accuracy.txt'
STD_ERROR_FILE = '/std_error.txt'

################ PLOT ###################
BASE_PLOT_PATH = '/home/pooja/Research_Project/Shared/Plot'
PERM_DECAY_PLOT_PATH = BASE_PLOT_PATH + '/Perm_decay'

ENERGY_PLOT = '/energy.png'
ENERGY_ELTP_PLOT = '/energy_eLTP.png'
ENERGY_LLTP_PLOT = '/energy_lLTP.png'
ERROR_PLOT = '/error.png'
EPOCH_PLOT = '/epoch.png'
PATTERNS_PLOT = '/patterns.png'
PATTERNS_SLOPE_PLOT = '/slope.png'
ACCURACY_PLOT = '/accuracy.png'
EPOCH_UPDATES_PLOT = '/epoch_updates.png'
ENERGY_UPDATES_PLOT = '/energy_updates.png'