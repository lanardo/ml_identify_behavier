import sys
sys.path.insert(0, "../common")

from model_checker import *

DATA_PATH = '../../../data/full/'
RESULT_PATH = '../output/'

check_model(DATA_PATH + 'inputs.ods', DATA_PATH + 'purpose_params.ods', 31, None, 'results_learning.xls')
