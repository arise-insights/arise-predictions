import argparse

from typing import Sequence

"""
This module parses command line arguments and passes the information to the main execution program
"""

cmd_args = "test123"


def parse_args(argv: Sequence[str] = None):
    """
    Parses the command line arguments and returns the parsed arguments.

    :param argv: Optional sequence of command-line arguments. Allows unit tests
        to pretend that a certain set of command-line arguments have been
        supplied. Will be ignored if not set.
    :type argv: Sequence
    """
    # Create the argument parser object
    parser = argparse.ArgumentParser(
        description='Command-line argument parser')
    parser.add_argument('--loglevel', '-l', default='debug',
                        help='log level', nargs='?', type=str)
    parser.add_argument('--num-jobs', default=-1,
                        help='number of cores to use. default is all', nargs='?', type=int)
    subparsers = parser.add_subparsers(
        help='commands help', dest='command')

    # parser for the preprocess command
    parser_preprocess = subparsers.add_parser(
        'preprocess', help='preprocess help')
    parser_preprocess.add_argument(
        '--input-path', default='examples/MLCommons', help='input path', nargs='?', type=str)
    parser_preprocess.add_argument(
        '--reread-history', help='reread history', action='store_true')
    parser_preprocess.add_argument(
        '--job-file', help='new job file full path for relevant history', nargs='?', type=str)
    parser_preprocess.add_argument(
        '--input-file', help='input data file name, in case only one of the files in input path should be used',
        nargs='?', type=str)
    parser_preprocess.add_argument(
        '--ignore-metadata', help='ignore metadata if exists', action='store_true')

    # parser for analyze jobs command
    parser_analyze_jobs = subparsers.add_parser(
        'analyze-jobs', help='analyze jobs help')
    parser_analyze_jobs.add_argument(
        '--input-path', default='examples/MLCommons', help='input path', nargs='?', type=str)
    parser_analyze_jobs.add_argument(
        '--reread-history', help='reread history', action='store_true')
    parser_analyze_jobs.add_argument(
        '--job-file', help='new job file full path for relevant history', nargs='?', type=str)
    parser_analyze_jobs.add_argument(
        '--input-file', help='input data file name, in case only one of the files in input path should be used',
        nargs='?', type=str)
    parser_analyze_jobs.add_argument(
        '--job-id-column', help='column name representing job id column. If not given, custom job name will be used '
                                'instead.',
        required=False, type=str)
    parser_analyze_jobs.add_argument(
        '--custom-job-name', help='job name to be used in job id column in case none is specified in job id column. '
                                  'If not given, default job name will be used.',
        required=False, type=str)
    parser_analyze_jobs.add_argument(
        '--ignore-metadata', help='ignore metadata if exists', action='store_true')

    # parser for auto build models command
    parser_auto_build_models = subparsers.add_parser(
        'auto-build-models', help='auto build models help')
    parser_auto_build_models.add_argument(
        '--input-path', default='examples/MLCommons', help='input path', nargs='?', type=str)
    parser_auto_build_models.add_argument(
        '--reread-history', help='reread history', action='store_true')
    parser_auto_build_models.add_argument(
        '--job-file', help='new job file full path for relevant history', nargs='?', type=str)
    parser_auto_build_models.add_argument(
        '--input-file', help='input data file name, in case only one of the files in input path should be used',
        nargs='?', type=str)
    parser_auto_build_models.add_argument(
        '--config-file', default='config/default-auto-model-search-config.yaml', 
        help="path to config file defining estimators and parameter search space",
        required=False, type=str)
    parser_auto_build_models.add_argument(
        '--leave-one-out-cv', help='leave one group out cross validation. Provide list of feature names separated'
                                   ' by commas', required=False, type=str)
    parser_auto_build_models.add_argument(
        '--randomized-hpo', help='Use random sampling hyperparameter optimization', action='store_true')
    parser_auto_build_models.add_argument(
        '--random-iterations', help='Number of sampling iterations to perform for each model hyper-parameter space',
        required=False, type=int)
    parser_auto_build_models.add_argument(
        '--feature-column', help='Name of the feature to extrapolate on',
        required=False, type=str)
    parser_auto_build_models.add_argument(
        '--low-threshold', help='exclude feature values less than or equal to the threshold',
        required=False, type=int)
    parser_auto_build_models.add_argument(
        '--high-threshold', help='exclude feature values greater than or equal to the threshold',
        required=False, type=int)
    parser_auto_build_models.add_argument(
        '--ignore-metadata', help='ignore metadata if exists', action='store_true')
    parser_auto_build_models.add_argument(
        '--single-output-file', help='package output into a single file', action='store_true')

    # parser for demo-predict command
    parser_demo_predict = subparsers.add_parser(
        'demo-predict', help='demo predict help')
    parser_demo_predict.add_argument(
        '--input-path', default='examples/MLCommons', help='input path', nargs='?', type=str)
    parser_demo_predict.add_argument(
        '--reread-history', help='reread history', action='store_true')
    parser_demo_predict.add_argument(
        '--input-file', help='input data file name, in case only one of the files in input path should be used',
        nargs='?', type=str)
    parser_demo_predict.add_argument(
        '--config-file', default='config/example-demo-mlcommons-config.yaml',
        help="path to config file defining estimators, target variables, and input search space",
        required=False, type=str)
    parser_demo_predict.add_argument(
        '--model-path', 
        help="path to serialized estimators", required=True, type=str)
    parser_demo_predict.add_argument(
        '--ignore-metadata', help='ignore metadata if exists', action='store_true')

    # parser for predict command
    parser_predict = subparsers.add_parser('predict', help='predict help')
    parser_predict.add_argument(
        '--input-path', default='', help='input path', nargs='?', type=str)
    parser_predict.add_argument(
        '--config-file', default='config/example-demo-mlcommons-config.yaml',
        help="path to config file defining estimators, target variables, and input search space",
        required=False, type=str)
    parser_predict.add_argument(
        '--model-path', help="path to serialized estimators", required=True, type=str)
    parser_predict.add_argument(
        '--ignore-metadata', help='ignore metadata if exists', action='store_true')

    # parser for data-predict command
    parser_data_predict = subparsers.add_parser(
        'data-predict', help='data predict help')
    parser_data_predict.add_argument(
        '--original-data-file', help='file containing the original data', nargs='?', required=False, type=str)
    parser_data_predict.add_argument(
        '--prediction-data-file', help='file containing the prediction data', required=True, nargs='?', type=str)
    parser_data_predict.add_argument(
        '--delta-only', help='predict only for prediction data not in original data', action='store_true')
    parser_data_predict.add_argument(
        '--input-path', help='input path', nargs='?', type=str)
    parser_data_predict.add_argument(
        '--config-file', default='config/example-demo-mlcommons-config.yaml',
        help="path to config file defining estimators, target variables, and input search space", type=str)
    parser_data_predict.add_argument(
        '--model-path', help="path to serialized estimators", required=True, type=str)

    # Parse the command line arguments
    global cmd_args
    if argv: 
        cmd_args = parser.parse_args(argv)
    else:
        cmd_args = parser.parse_args()
    return True


def get_args():
    global cmd_args
    return cmd_args
