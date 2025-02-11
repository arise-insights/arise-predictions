JOB_INPUTS_FIELD_NAME = 'job-metadata-inputs'
JOB_OUTPUTS_FIELD_NAME = 'job-metadata-outputs'
JOB_START_TIME_FIELD_NAME = 'start-time-field-name'
JOB_END_TIME_FIELD_NAME = 'end-time-field-name'
JOB_START_TIME_DEFAULT_FIELD_NAME = 'start_time'
JOB_END_TIME_DEFAULT_FIELD_NAME = 'end_time'
JOB_DURATION_FIELD_NAME = 'duration'
JOB_HISTORY_FILE_NAME = "data_execution_history"
JOB_METADATA_FILE_NAME = "data_execution_metadata"
DEFAULT_CORR_THRESHOLD = 0.85
KS_TEST_SIGNIFICANCE_LEVEL = 0.05
JOB_PARSER_CLASS_NAME_FIELD = 'job-parser-class-name'
METADATA_PARSER_CLASS_NAME_FIELD = 'metadata-parser-class-name'
JOB_ENTRY_FILTER_FIELD = 'job-entry-filter'
JOB_ENTRY_FILTER_NAME_COL = 'name'
JOB_ENTRY_FILTER_VALUES_COL = 'excluded_values'
JOB_ENTRY_FILTER_KEEP_COL = 'keep_input'
DUMMY_VARS_PREFIX = 'dummy_input_'
JOB_INPUTS_FEATURE_ENGINEERING = 'job-metadata-fe'
JOB_DATA_DIR = "data"
JOB_METADATA_DIR = "metadata"
JOB_SPEC_FILE_NAME = "job_spec.yaml"


# job data analysis
JOB_ANALYSIS_PATH = "job-analysis"
DEFAULT_JOB_ID_COLUMN = "job_name"
DEFAULT_JOB_NAME = "single-job"
NUMERICS = ["int16", "int32", "int64", "float16", "float32", "float64"]

# matplotlib
FONT_SMALL_SIZE = 8
FONT_MEDIUM_SIZE = 10

# auto model
SEED = 42
AM_CONFIG_ESTIMATORS = "estimators"
AM_CONFIG_NAME = "name"
AM_CONFIG_CLASS_NAME = "class_name"
AM_CONFIG_LINEAR = "linear"
AM_CONFIG_PARAMETERS = "parameters"
AM_CONFIG_LINEARTREE = "lineartree"
AM_ESTIMATORS_NO_SEED = ["LinearRegression", "SVR", "DecisionTreeRegressor",
                         "KNeighborsRegressor"]
AM_ESTIMATORS_CATBOOST = ["CatBoostRegressor"]
AM_DEFAULT_METRIC = "neg_mean_absolute_percentage_error"
AM_OUTPUT_PATH_SUFFIX = "ARISE-auto-models"
AM_NUM_JOBS = "num_jobs"
AM_RANKINGS_FILE = "all-star-rankings.csv"
AM_META_LEARNER_PREFIX = "meta-learner-"
AM_DEFAULT_N_ITER_RANDOM_HPO = 50

# auto model column headers
AM_COL_ESTIMATOR = "estimator"
AM_COL_LINEAR = "linear"
AM_COL_TARGET = "target_variable"
AM_COL_TARGET_RANGE = "target_variable_range"
AM_COL_RANK_MAPE = "rank_MAPE"
AM_COL_RANK_NRMSE_MAXMIN = "rank_NRMSE_maxmin"
AM_COL_RANK_R2 = "rank_R2"

# auto model performance metrics
# TODO consider moving to metrics.py
AM_METRIC_MAPE_MEAN = "mean_test_neg_mean_absolute_percentage_error"
AM_METRIC_NRMSE_MEAN = "mean_test_neg_normalized_root_mean_squared_error(minmax)"
AM_METRIC_R2_MEAN = "mean_test_r2"

# predict
PRED_OUTPUT_PATH_SUFFIX = "ARISE-predictions"
PRED_INPUT_SPACE_FILE = "input-space.csv"
PRED_ALL_PREDICTIONS_FILE = "all-predictions.csv"
PRED_GROUND_TRUTH_FILE = "predictions-with-ground-truth.csv"
PRED_ORIGINAL_TRUTH_FILE = "original-data.csv"
PRED_CONFIG_FIXED = "fixed_values"
PRED_CONFIG_VARIABLE = "variable_values"
PRED_CONFIG_DATA = "data_values"
PRED_CONFIG_DATA_INPUT = "input_variable"
PRED_CONFIG_DATA_VALUES = "values"
PRED_CONFIG_DATA_ALL = "all"
PRED_CONFIG_DATA_MIN_MAX = "min_max"
PRED_CONFIG_DATA_EXCLUDE = "exclude"
PRED_CONFIG_INTERPOLATION = "interpolation_values"
PRED_CONFIG_ESTIMATORS = "estimators"
PRED_CONFIG_TARGET_VAR = "target_variable"
PRED_CONFIG_GREATER_BETTER = "greater_is_better"
PRED_CONFIG_ESTIMATOR_FILE = "estimator_file"

# extrapolation
EXTRA_TEST_FILE = "extrapolation-test-data.csv"
