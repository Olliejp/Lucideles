""" RAW data """

# raw data inputs
ILL_RAW_DATA = "/Users/oliverpaul/Data_Science/idiap/lucideles/lucideles_repo/project_2/room_1/inputs/WPI"
DGP_RAW_DATA = "/Users/oliverpaul/Data_Science/idiap/lucideles/lucideles_repo/project_2/room_1/inputs/DGP"
WEATHER_DATA = "/Users/oliverpaul/Data_Science/idiap/lucideles/lucideles_repo/project_2/room_1/inputs/weather"

""" Input data """

# data path
DATA_PATH = "/Users/oliverpaul/Data_Science/idiap/lucideles/lucideles_repo/project_2/room_1/inputs/"
# data with no transformed azimuth
DATA_RAW = "/Users/oliverpaul/Data_Science/idiap/lucideles/lucideles_repo/project_2/room_1/inputs/data.csv"
# data with transformed azimuth
DATA = "/Users/oliverpaul/Data_Science/idiap/lucideles/lucideles_repo/project_2/room_1/inputs/data_transform.csv"
#training and testing data
TRAIN_ILL = "/Users/oliverpaul/Data_Science/idiap/lucideles/lucideles_repo/project_2/room_1/inputs/train_ill.csv"
TRAIN_DGP = "/Users/oliverpaul/Data_Science/idiap/lucideles/lucideles_repo/project_2/room_1/inputs/train_dgp.csv"
TEST_ILL = "/Users/oliverpaul/Data_Science/idiap/lucideles/lucideles_repo/project_2/room_1/inputs/test_ill.csv"
TEST_DGP = "/Users/oliverpaul/Data_Science/idiap/lucideles/lucideles_repo/project_2/room_1/inputs/test_dgp.csv"
# training data with folds column
TRAIN_ILL_WITH_FOLDS = "/Users/oliverpaul/Data_Science/idiap/lucideles/lucideles_repo/project_2/room_1/inputs/train_ill_folds.csv"
TRAIN_DGP_WITH_FOLDS = "/Users/oliverpaul/Data_Science/idiap/lucideles/lucideles_repo/project_2/room_1/inputs/train_dgp_folds.csv"


""" Outputs """

OUTPUTS_PATH = "/Users/oliverpaul/Data_Science/idiap/lucideles/lucideles_repo/project_2/room_1/outputs/"
#numpy array of shuffle order for full dataset
SHUFFLE_ORDER = "/Users/oliverpaul/Data_Science/idiap/lucideles/lucideles_repo/project_2/room_1/outputs/shuff_order.npy"

""" PARAMS """

DROP_COLUMNS = ['month', 'day', 'hour']

NORM_COLUMNS = ['altitude', 
                'ibn', 
                'idh', 
                'blind_angle_n', 
                'blind_angle_s',]

FEATURES = ['altitude', 
            'ibn', 
            'idh', 
            'reference',
            'blind_angle_n', 
            'blind_angle_s',
            'azimuth_sin',
            'azimuth_cos']

