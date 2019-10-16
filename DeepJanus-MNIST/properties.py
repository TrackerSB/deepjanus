# GA Setup
POPSIZE = 100
NGEN = 300

# Mutation Hyperparameters
# range of the mutation
MUTLOWERBOUND = 0.01
MUTUPPERBOUND = 0.6

# Reseeding Hyperparameters
# extent of the reseeding operator
RESEEDUPPERBOUND = 10

K_SD = 0.001

# K-nearest
K = 1

# Archive configuration
ARCHIVE_THRESHOLD = 1.5

# Dataset
EXPECTED_LABEL = 5

#------- NOT TUNING ----------

# mutation operator probability
MUTOPPROB = 0.5
MUTOFPROB = 0.5

IMG_SIZE = 28
num_classes = 10


INITIALPOP = 'seeded'
#INITIALPOP = 'random'


GENERATE_ONE_ONLY = False

# Directories
PATH = "vectorized_images_GA"
TESTSOURCEPATH = "source_images_GA"
TRAINSOURCEPATH = "source_images_trainset"

#MODEL = 'models/cnnClassifier_lowLR.h5'
MODEL = 'models/cnnClassifier.h5'

ORIGINAL_SEEDS = "first_generation"
RESULTS_PATH = 'results'