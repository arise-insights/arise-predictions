import pandas as pd
import os
from arise_predictions.cmd.cmd import parse_args, get_args
from arise_predictions.job_statistics.analyze_jobs import analyze_job_data
from arise_predictions.auto_model.build_models import auto_build_models, get_estimators_config
from arise_predictions.perform_predict.predict import demo_predict, get_predict_config
from arise_predictions.utils import constants, utils
from arise_predictions.preprocessing import job_parser
from arise_predictions.sdk import execute_auto_build_models
import logging

logger = logging.getLogger(__name__)

def execute_analyze_jobs():
    loaded_job_spec = load_spec(get_args().input_path, get_args().job_spec_file_name)
    outputs = sorted(list(loaded_job_spec[1]))

    # processing history ( if not done in the past )
    history_data, history_file = execute_preprocess(loaded_job_spec, get_args().input_path)

    if history_data is None or history_data.empty:
        logging.error(("No historical data could be retrieved from given" 
                       " location {}").format(get_args().input_path))
    else:
        logging.info("Invoking job analysis")
        analyze_job_data(raw_data=history_data, job_id_column=get_args().job_id_column,
                         custom_job_name=get_args().custom_job_name,
                         output_path=os.path.join(get_args().input_path,
                                                  constants.JOB_ANALYSIS_PATH),
                         target_variables=outputs)

def execute_demo_predict():
    loaded_job_spec = load_spec(get_args().input_path, get_args().job_spec_file_name)

    # processing history ( if not done in the past )
    history_data, history_file = execute_preprocess(loaded_job_spec, get_args().input_path)

    if history_data is None or history_data.empty:
        logging.error(("No historical data could be retrieved from given" 
                       " location {}").format(get_args().input_path))
    else:
        logging.info("Invoking demo predict")
        demo_predict(
            original_data=history_data,
            config=get_predict_config(get_args().config_file),
            estimator_path=get_args().model_path,
            feature_engineering=None if get_args().ignore_metadata else loaded_job_spec[6],
            metadata_parser_class_name=loaded_job_spec[7],
            metadata_path=get_args().input_path,
            output_path=os.path.join(get_args().input_path, constants.PRED_OUTPUT_PATH_SUFFIX))

def main():
    if not parse_args():
        global logger
        logger.error('Failed to parse command line arguments')
        exit(1)
    level = logging.getLevelName(get_args().loglevel.upper())
    print("level: %d" % level)
    print("cmd_args: %s" % get_args())

    logger = logging.getLogger()
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=level)

    input_path = get_args().input_path
    # According to the selected command, call the appropriate function
    if get_args().command == 'preprocess':
        execute_preprocess(load_spec(input_path, get_args().job_spec_file_name), args)
    elif get_args().command == 'analyze-jobs':
        execute_analyze_jobs()
    elif get_args().command == 'auto-build-models':
        execute_auto_build_models(get_args())
    elif get_args().command == 'demo-predict':
        execute_demo_predict()
    elif get_args().command == 'predict':
        execute_predict(get_args())
    else:
        logger.error('Invalid command!')
        logger.info(
            'For development purpose we will execute the get-stats command')
        logger.info('This will be removed in production')
        from arise_predictions.cmd.cmd import cmd_args
        cmd_args.command = "get-stats"
        cmd_args.input_path = "examples/MLCommons"
        cmd_args.reread_history = False
        execute_analyze_jobs()


if __name__ == '__main__':
    main()
