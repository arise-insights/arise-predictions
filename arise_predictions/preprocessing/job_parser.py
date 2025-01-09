
import yaml
import os
import pandas as pd
import numpy as np

from arise_predictions.utils import utils, constants
from arise_predictions.preprocessing import custom_job_parser
import logging
from typing import List

logger = logging.getLogger(__name__)

"""
This module parsers a job spec and collects its historical executions from different representations
into CSV format based on the given spec. All spec fields are assumed to be numeric.
"""


def parse_job_spec(job_spec_file):
    # Reading the job spec file
    with open(job_spec_file, "r") as file:
        try:
            loaded_spec = dict(yaml.safe_load(file))
        except yaml.YAMLError as exc:
            logger.error(exc)
            return None

    # Making sure that all required fields are available in the spec
    for field in [constants.JOB_INPUTS_FIELD_NAME, constants.JOB_OUTPUTS_FIELD_NAME]:
        if field not in loaded_spec:
            logger.error("Missing key: {}".format(field))
            return None

    # Reading start and end time field names, using defaults if not provided
    if constants.JOB_START_TIME_FIELD_NAME in loaded_spec:
        start_time_field_name = loaded_spec[constants.JOB_START_TIME_FIELD_NAME]
    else:
        start_time_field_name = constants.JOB_START_TIME_DEFAULT_FIELD_NAME

    if constants.JOB_END_TIME_FIELD_NAME in loaded_spec:
        end_time_field_name = loaded_spec[constants.JOB_END_TIME_FIELD_NAME]
    else:
        end_time_field_name = constants.JOB_END_TIME_DEFAULT_FIELD_NAME

    if constants.JOB_PARSER_CLASS_NAME_FIELD in loaded_spec:
        job_parser_class_name = loaded_spec[constants.JOB_PARSER_CLASS_NAME_FIELD]
    else:
        job_parser_class_name = None

    if constants.JOB_ENTRY_FILTER_FIELD in loaded_spec:
        job_entry_filter = loaded_spec[constants.JOB_ENTRY_FILTER_FIELD]
    else:
        job_entry_filter = dict()

    if constants.JOB_INPUTS_FEATURE_ENGINEERING in loaded_spec:
        job_input_feature_engineering = loaded_spec[constants.JOB_INPUTS_FEATURE_ENGINEERING]
    else:
        job_input_feature_engineering = dict()

    if constants.METADATA_PARSER_CLASS_NAME_FIELD in loaded_spec:
        metadata_parser_class_name = loaded_spec[constants.METADATA_PARSER_CLASS_NAME_FIELD]
    else:
        metadata_parser_class_name = None

    return (set(loaded_spec[constants.JOB_INPUTS_FIELD_NAME]), set(loaded_spec[constants.JOB_OUTPUTS_FIELD_NAME]),
            start_time_field_name, end_time_field_name, job_parser_class_name, job_entry_filter,
            job_input_feature_engineering, metadata_parser_class_name)

# This function computes duration fo the job based on start and end time
# It needs to work on each job row separately, because unfortunately different rows report time in different formats


def get_job_duration_from_json(job_data, start_time_field_name, end_time_field_name):

    start_time = utils.find_item(job_data, start_time_field_name)
    end_time = utils.find_item(job_data, end_time_field_name)

    if not start_time or not end_time:
        raise Exception("Both {} and {} should be in historical executions".format(start_time_field_name,
                                                                                   end_time_field_name))

    return utils.get_duration(start_time, end_time)


def process_single_file(filename, columns, class_inst, job_parser_obj):

    if os.path.isfile(filename):
        if filename.endswith(".csv"):
            if class_inst:
                return getattr(class_inst, "get_history_csv")(filename)
            else:
                return pd.read_csv(filename)[columns]
        elif filename.endswith(".json"):
            if class_inst:
                return getattr(class_inst, "get_history_json")(filename)
            else:
                return job_parser_obj.get_history_json(filename)
        else:
            logger.warning("file format of {} not supported, skipping".format(filename))
            return None

    logger.warning("{} is not a file, skipping".format(filename))
    return None


def collect_jobs_history(data_dir, output_path, job_inputs, job_outputs, start_time_field_name,
                         end_time_field_name, input_file=None, job_parser_class_name=None, job_entry_filter={},
                         feature_engineering=None, metadata_parser_class_name=None, metadata_path=None):

    """
    Collects historical executions of jobs into a CSV file according to given job input and output spec. Replaces
    start and end time columns with duration column.

    :param data_dir:
    :param output_path:
    :param job_inputs:
    :param job_outputs:
    :param start_time_field_name:
    :param end_time_field_name:
    :param input_file:
    :param job_parser_class_name:
    :param job_entry_filter
    :param feature_engineering: dict of feature to augment and their corresponding csv file
    :param metadata_parser_class_name class name of the parser for metadata info for feature engineering
    :param metadata_path:path to metadata files
    :return: a dataframe containing the historical data and a file name to which the data was stored. None if parsing
    failed.
    """

    columns_with_derived = utils.adjust_columns_with_duration(job_inputs + job_outputs, start_time_field_name,
                                                              end_time_field_name)

    # add columns to be filtered by (to be removed at the end of processing)
    filter_columns = list(job_entry_filter.keys())
    columns_with_derived = columns_with_derived + filter_columns

    df = pd.DataFrame(columns=columns_with_derived)

    if not os.path.exists(data_dir):
        logger.error("Data directory {} does not exist".format(data_dir))
        return None, None

    if not os.path.exists(output_path):
        logger.error("output path {} does not exist".format(output_path))
        return None, None

    job_parser_obj = None
    class_inst = None

    if job_parser_class_name:
        logger.info("Parsing jobs history with class {}".format(job_parser_class_name))
        estimation_module = __import__('preprocessing.custom_job_parser', fromlist=[job_parser_class_name])
        class_inst = getattr(estimation_module, job_parser_class_name)(columns_with_derived, start_time_field_name,
                                                                       end_time_field_name)
    else:
        job_parser_obj = custom_job_parser.DefaultJsonJobParser(columns_with_derived, start_time_field_name,
                                                                end_time_field_name)

    if input_file is not None:
        df = process_single_file(os.path.join(data_dir, input_file), columns_with_derived, class_inst,
                                 job_parser_obj)
    else:
        for root, subdirs, files in os.walk(data_dir):
            for filename in files:
                # each metadata file may contain more than one job
                new_history_df = process_single_file(os.path.join(root, filename), columns_with_derived, class_inst,
                                                     job_parser_obj)
                if new_history_df is not None:
                    new_history_df.reset_index(drop=True, inplace=True)
                    if df.empty:
                        df = new_history_df
                    else:
                        df.reset_index(drop=True, inplace=True)
                        df = pd.concat([df, new_history_df])
    if df.empty:
        logger.error("No execution history found in {}".format(data_dir))
        return None, None
    else:
        if job_entry_filter:
            for key, value in job_entry_filter.items():
                df = df[df[key] != value]
                df = df.drop(key, axis=1)
        logger.info("Found {:d} executions in history".format(len(df)))

    collect_and_persist_data_metadata(df, job_inputs, job_outputs, output_path)

    if feature_engineering is not None:
        df = utils.add_feature_engineering(metadata_path, df, feature_engineering, metadata_parser_class_name)

    output_file = os.path.join(output_path, constants.JOB_HISTORY_FILE_NAME + ".csv")

    df.to_csv(output_file, index=False)
    return df, output_file


def collect_and_persist_data_metadata(df: pd.DataFrame, inputs: List[str], outputs: List[str], output_path: str):

    data_metadata = dict()

    data_metadata['inputs'] = inputs
    data_metadata['outputs'] = outputs

    categorical_features = utils.get_categorical_features(df)

    for job_input in inputs:

        input_metadata = dict()
        input_metadata['type'] = 'categorical' if job_input in categorical_features else 'numeric'
        if input_metadata['type'] == 'categorical':
            input_metadata['values'] = df[job_input].unique().tolist()
        else:
            input_min = df[job_input].min()
            input_max = df[job_input].max()
            input_metadata['min'] = float(input_min) if isinstance(input_min, np.floating) else int(input_min)
            input_metadata['max'] = float(input_max) if isinstance(input_max, np.floating) else int(input_max)

        data_metadata[job_input] = input_metadata

    output_file = os.path.join(output_path, constants.JOB_METADATA_FILE_NAME + ".yaml")

    with open(output_file, 'w') as out:
        yaml.dump(data_metadata, out)
