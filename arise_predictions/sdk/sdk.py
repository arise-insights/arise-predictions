import os

from arise_predictions.utils import constants, utils
from arise_predictions.auto_model.build_models import auto_build_models, get_estimators_config
from arise_predictions.preprocessing import job_parser
from arise_predictions.cmd.cmd import process_args
import logging

logger = logging.getLogger(__name__)

def load_spec(input_path, spec_file_name):
    # analyzing job spec file
    job_spec_file = os.path.join(input_path, spec_file_name)
    logger.info('Analyzing job spec file: %s', job_spec_file)
    loaded_job_spec = job_parser.parse_job_spec(job_spec_file)
    if not loaded_job_spec:
        logger.error("Failed to load job spec")
        raise Exception
    return loaded_job_spec

def get_history(history_file, inputs, outputs, start_time_field_name, end_time_field_name, job_parser_class_name, job_entry_filter, feature_engineering, metadata_parser_class_name, args):

    if os.path.exists(history_file) and not args.reread_history:
        logging.info("using already processed history")
        history_data = pd.read_csv(history_file)
        history_data = history_data[utils.adjust_columns_with_duration(history_data.columns.values.tolist(),
                                                                       start_time_field_name,
                                                                       end_time_field_name)]
    else:
        logging.info("processing historical jobs")
        history_data, history_file = job_parser.collect_jobs_history(
            args.input_path + "/" + constants.JOB_DATA_DIR, args.input_path, inputs, outputs,
            start_time_field_name, end_time_field_name, args.input_file, job_parser_class_name, job_entry_filter,
            None if args.ignore_metadata else feature_engineering, metadata_parser_class_name, args.input_path)
    return history_data, history_file

def get_base_args(fun: str):
    return process_args([fun])

def execute_preprocess(job_spec, args):
    inputs = sorted(list(job_spec[0]))
    outputs = sorted(list(job_spec[1]))
    start_time_field_name = job_spec[2]
    end_time_field_name = job_spec[3]
    job_parser_class_name = job_spec[4]
    job_entry_filter = job_spec[5]
    feature_engineering = job_spec[6] if len(job_spec) > 6 else None
    metadata_parser_class_name = job_spec[7] if len(job_spec) > 7 else None

    # processing history ( if not done in the past )
    analyzed_history_file = os.path.join(args.input_path, constants.JOB_HISTORY_FILE_NAME + ".csv")

    return get_history(analyzed_history_file, inputs, outputs, start_time_field_name, end_time_field_name, job_parser_class_name, job_entry_filter, feature_engineering, metadata_parser_class_name, args)

def execute_auto_build_models(args):
    loaded_job_spec = load_spec(args.input_path, args.job_spec_file_name)
    outputs = sorted(list(loaded_job_spec[1])) 

    # processing history ( if not done in the past )
    history_data, history_file = execute_preprocess(loaded_job_spec, args)

    if history_data is None or history_data.empty:
        logging.error(("No historical data could be retrieved from given" 
                       " location {}").format(args.input_path))
    else:
        logging.info("Invoking auto model search and build")
        auto_build_models(raw_data=history_data,
                          config=get_estimators_config(config_file=args.config_file,
                                                       num_jobs=args.num_jobs),
                          target_variables=outputs,
                          output_path=os.path.join(args.input_path, constants.AM_OUTPUT_PATH_SUFFIX),
                          leave_one_out_cv=args.leave_one_out_cv,
                          feature_col=args.feature_column,
                          low_threshold=args.low_threshold,
                          high_threshold=args.high_threshold,
                          single_output_file=args.single_output_file)
