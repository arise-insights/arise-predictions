import shutil

import pandas as pd
import os
from arise_predictions.cmd.cmd import parse_args, get_args
from arise_predictions.job_statistics.analyze_jobs import analyze_job_data
from arise_predictions.auto_model.build_models import auto_build_models, get_estimators_config
from arise_predictions.perform_predict.predict import demo_predict, data_predict, get_predict_config
from arise_predictions.utils import constants, utils
from arise_predictions.preprocessing import job_parser
import logging

logger = logging.getLogger(__name__)


def load_spec(spec_path):
    # analyzing job spec file

    spec_folder = utils.get_unpacked_path(spec_path)

    job_spec_file = os.path.join(spec_folder, constants.JOB_SPEC_FILE_NAME)
    logger.info('Analyzing job spec file: %s', job_spec_file)
    loaded_job_spec = job_parser.parse_job_spec(job_spec_file)
    if not loaded_job_spec:
        logger.error("Failed to load job spec")
        raise Exception
    return loaded_job_spec


def execute_preprocess(spec_path):

    job_spec = load_spec(spec_path)

    inputs = sorted(list(job_spec[0]))
    outputs = sorted(list(job_spec[1]))
    start_time_field_name = job_spec[2]
    end_time_field_name = job_spec[3]
    job_parser_class_name = job_spec[4]
    job_entry_filter = job_spec[5]
    feature_engineering = job_spec[6] if len(job_spec) > 6 else None
    metadata_parser_class_name = job_spec[7] if len(job_spec) > 7 else None

    # processing history ( if not done in the past )
    analyzed_history_file = os.path.join(
        get_args().input_path, constants.JOB_HISTORY_FILE_NAME + ".csv")
    return get_history(analyzed_history_file, inputs, outputs, start_time_field_name, end_time_field_name,
                       job_parser_class_name, job_entry_filter, feature_engineering, metadata_parser_class_name)


def execute_analyze_jobs():

    # processing history ( if not done in the past )
    history_data, history_file = execute_preprocess(get_args().input_path)

    if history_data is None or history_data.empty:
        logging.error(("No historical data could be retrieved from given" 
                       " location {}").format(get_args().input_path))
    else:
        loaded_job_spec = load_spec(get_args().input_path)
        outputs = sorted(list(loaded_job_spec[1]))
        logging.info("Invoking job analysis")
        analyze_job_data(raw_data=history_data, job_id_column=get_args().job_id_column,
                         custom_job_name=get_args().custom_job_name,
                         output_path=os.path.join(get_args().input_path,
                                                  constants.JOB_ANALYSIS_PATH),
                         target_variables=outputs)


def execute_auto_build_models():

    # processing history ( if not done in the past )
    history_data, history_file = execute_preprocess(get_args().input_path)

    if history_data is None or history_data.empty:
        logging.error(("No historical data could be retrieved from given" 
                       " location {}").format(get_args().input_path))
    else:
        # First copy job spec file and metadata dir (if exists) to output path,
        # so they are available later for prediction. This is done before building the models,
        # because afterwards output folder may be zipped and deleted.
        output_path = os.path.join(get_args().input_path, constants.AM_OUTPUT_PATH_SUFFIX)
        utils.mkdirs(output_path)
        shutil.copy(os.path.join(get_args().input_path, constants.JOB_SPEC_FILE_NAME), output_path)
        loaded_job_spec = load_spec(get_args().input_path)
        outputs = sorted(list(loaded_job_spec[1]))
        feature_engineering = loaded_job_spec[6] if len(loaded_job_spec) > 6 else None
        if feature_engineering and not get_args().ignore_metadata:
            shutil.copytree(os.path.join(get_args().input_path, constants.JOB_METADATA_DIR),
                            os.path.join(output_path, constants.JOB_METADATA_DIR),
                            dirs_exist_ok=True)
        logging.info("Invoking auto model search and build")
        auto_build_models(raw_data=history_data,
                          config=get_estimators_config(config_file=get_args().config_file,
                                                       num_jobs=get_args().num_jobs),
                          target_variables=outputs,
                          output_path=output_path,
                          leave_one_out_cv=get_args().leave_one_out_cv,
                          feature_col=get_args().feature_column,
                          low_threshold=get_args().low_threshold,
                          high_threshold=get_args().high_threshold,
                          single_output_file=get_args().single_output_file,
                          randomized_hpo=get_args().randomized_hpo,
                          n_random_iter=get_args().random_iterations)


def execute_demo_predict():

    # processing history ( if not done in the past )
    history_data, history_file = execute_preprocess(get_args().model_path)

    if history_data is None or history_data.empty:
        logging.error(("No historical data could be retrieved from given" 
                       " location {}").format(get_args().input_path))
    else:
        loaded_job_spec = load_spec(get_args().model_path)
        logging.info("Invoking demo predict")
        demo_predict(
            original_data=history_data,
            config=get_predict_config(get_args().config_file),
            estimator_path=get_args().model_path,
            feature_engineering=None if get_args().ignore_metadata else loaded_job_spec[6],
            metadata_parser_class_name=loaded_job_spec[7],
            metadata_path=get_args().model_path,
            output_path=os.path.join(get_args().input_path, constants.PRED_OUTPUT_PATH_SUFFIX))


def execute_predict():
    loaded_job_spec = load_spec(get_args().model_path)

    logging.info("Invoking predict")
    demo_predict(
        original_data=None,
        config=get_predict_config(get_args().config_file),
        estimator_path=get_args().model_path,
        feature_engineering=None if get_args().ignore_metadata else loaded_job_spec[6],
        metadata_parser_class_name=loaded_job_spec[7],
        metadata_path=get_args().model_path,
        output_path=os.path.join(get_args().input_path, constants.PRED_OUTPUT_PATH_SUFFIX))


def execute_data_predict():
    loaded_job_spec = load_spec(get_args().model_path)
    input_path = get_args().input_path

    logging.info("Invoking data predict")
    data_predict(
        original_data=pd.read_csv(os.path.join(input_path, get_args().original_data_file)) if
        get_args().original_data_file else None,
        prediction_data=pd.read_csv(os.path.join(input_path, get_args().prediction_data_file)),
        estimator_path=get_args().model_path,
        estimators_config=get_predict_config(get_args().config_file).estimators,
        target_variables=loaded_job_spec[1],
        delta_only=get_args().delta_only,
        output_path=os.path.join(input_path, constants.PRED_OUTPUT_PATH_SUFFIX))


def get_history(history_file, inputs, outputs, start_time_field_name, end_time_field_name, job_parser_class_name,
                job_entry_filter, feature_engineering, metadata_parser_class_name):
    if os.path.exists(history_file) and not get_args().reread_history:
        logging.info("using already processed history")
        history_data = pd.read_csv(history_file)
        history_data = history_data[utils.adjust_columns_with_duration(history_data.columns.values.tolist(),
                                                                       start_time_field_name,
                                                                       end_time_field_name)]
    else:
        logging.info("processing historical jobs")
        history_data, history_file = job_parser.collect_jobs_history(
            os.path.join(get_args().input_path, constants.JOB_DATA_DIR), get_args().input_path, inputs, outputs,
            start_time_field_name, end_time_field_name, get_args().input_file, job_parser_class_name, job_entry_filter,
            None if get_args().ignore_metadata else feature_engineering, metadata_parser_class_name,
            get_args().input_path)
    return history_data, history_file


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

    # According to the selected command, call the appropriate function
    if get_args().command == 'preprocess':
        execute_preprocess(get_args().input_path)
    elif get_args().command == 'analyze-jobs':
        execute_analyze_jobs()
    elif get_args().command == 'auto-build-models':
        execute_auto_build_models()
    elif get_args().command == 'demo-predict':
        execute_demo_predict()
    elif get_args().command == 'predict':
        execute_predict()
    elif get_args().command == 'data-predict':
        execute_data_predict()
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
