from typing import Dict, Any
from dataclasses import dataclass
import logging
from itertools import product
import os
import sys
import glob
from functools import reduce
import shutil, zipfile
import yaml

import pandas as pd

from sklearn.metrics import mean_absolute_percentage_error

from arise_predictions.utils import utils, constants

logger = logging.getLogger(__name__)

"""
Performs predictions on input data using previously build estimators and
configuration defining space of input variables to explore. Processes historic
data so that it can create comparisons with ground truth as much as possible for
demo purposes.
"""


@dataclass
class PredictionInputSpace:

    fixed_values: list[Dict[str, Any]]
    data_values: list[Dict[Any, Any]]
    variable_values: list[Dict[str, Any]]
    interpolation_values: list[str]
    estimators: list[Dict[str, Any]]


def get_predict_config(config_file: str) -> PredictionInputSpace:
    logger.info(f"Reading YAML configuration: {config_file}")
    try:
        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f)
    except FileNotFoundError:
        raise ValueError(f"Configuration file not found: {config_file}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing configuration file: {e}")
    return get_predict_config_from_dict(config_dict)


def get_predict_config_from_dict(config_dict: dict[str, list[dict[Any, Any]]]) -> PredictionInputSpace:
    return PredictionInputSpace(config_dict.get(constants.PRED_CONFIG_FIXED, []),
                                config_dict.get(constants.PRED_CONFIG_DATA, []),
                                config_dict.get(constants.PRED_CONFIG_VARIABLE, []),
                                config_dict.get(constants.PRED_CONFIG_INTERPOLATION, []),
                                config_dict[constants.PRED_CONFIG_ESTIMATORS])


def _create_input_space(input_config: PredictionInputSpace,
                        original_data: pd.DataFrame,
                        output_path: str, feature_engineering: dict[str, str] = None,
                        metadata_parser_class_name: str = None, metadata_path: str = None) -> tuple[str, pd.DataFrame]:
    """
    Takes the configuration of the input space and creates a corresponding
    DataFrame to run predictions on.

    :param input_config: Parsed input feature space configuration
    :type input_config: PredictionInputSpace
    :param original_data: Original data used to create the model
    :type original_data: pd.DataFrame
    :param output_path: Path under which to store outputs of this script
    :type output_path: str
    :param feature_engineering: map from feature to its metadata features
    :type feature_engineering: dict[str, str]
    :param metadata_parser_class_name: class name of metadata features parser
    :type metadata_parser_class_name: str
    :param metadata_path: path to metadata files
    :type metadata_path: str
    :returns: Tuple containing path under which DataFrame representing input 
        feature space has been stored and the corresponding data frame
    :rtype: tuple[str, pd.DataFrame]
    """
    logger.info(f"Creating input feature space")
    fixed_values = {k: v for d in input_config.fixed_values for k, v in d.items()}
    data_values = {k[constants.PRED_CONFIG_DATA_INPUT]: _get_data_range_values(original_data, k[constants.PRED_CONFIG_DATA_INPUT], k)
                   for k in input_config.data_values} if original_data is not None and not original_data.empty else {}
    variable_values = {k: v for d in input_config.variable_values for k, v in d.items()}
    interpolation_values = {k: _get_data_range_missing_values(original_data, k) for k in
                            input_config.interpolation_values} if original_data is not None and not \
        original_data.empty else {}

    if variable_values.keys() & fixed_values.keys():
        logger.error("Cannot specify same variable name as variable values and fixed value")
        raise Exception

    if variable_values.keys() & interpolation_values.keys():
        logger.error("Cannot specify same variable name as variable values and interpolation values")
        raise Exception

    non_fixed_values = dict(data_values)
    for k, v in variable_values.items():
        non_fixed_values[k] = non_fixed_values[k] + v if k in data_values.keys() else v
    variable_dict = non_fixed_values | interpolation_values

    # create Cartesian product of variable values (much nicer than nested loops)
    combinations = list(product(*(variable_dict.values())))

    # create DF with resulting combinations
    input_space_df = pd.DataFrame(combinations, columns=variable_dict.keys())

    # add the fixed values to each row in that DF
    for var_name, val in fixed_values.items():
        input_space_df[var_name] = val

    if feature_engineering:
        input_space_df = utils.add_feature_engineering(metadata_path, input_space_df, feature_engineering,
                                                       metadata_parser_class_name)

    output_file_df = constants.PRED_INPUT_SPACE_FILE

    path = utils.write_df_to_csv(
        df=input_space_df,
        output_path=output_path,
        output_file=output_file_df)
    logger.info(f"Wrote input feature space to {path}")
    return path, input_space_df


def _get_data_range_missing_values(original_data: pd.DataFrame, var_name: str) -> list[str]:
    values = range(original_data[var_name].min() + 1, original_data[var_name].max())
    return [val for val in values if val not in original_data[var_name].tolist()]


def _get_data_range_values(original_data: pd.DataFrame, var_name: str, var_info: Dict[Any, Any]) -> list[str]:
    values = original_data[var_name].dropna().unique().tolist() if \
        var_info[constants.PRED_CONFIG_DATA_VALUES] == constants.PRED_CONFIG_DATA_ALL else \
        list(range(original_data[var_name].min(), original_data[var_name].max()+1))
    return values if constants.PRED_CONFIG_DATA_EXCLUDE not in var_info.keys() else \
        [val for val in values if val not in var_info[constants.PRED_CONFIG_DATA_EXCLUDE]]


def _get_highest_ranked_estimator(estimator_path: str, target_variable: str) -> str:
    ranking_file = os.path.join(estimator_path, constants.AM_RANKINGS_FILE)
    if not os.path.exists(ranking_file):
        logger.error("Failed to locate model ranking file")
        raise Exception
    ranking_df = pd.read_csv(ranking_file)
    ranking_df_var = ranking_df.loc[ranking_df[constants.AM_COL_TARGET] == target_variable]
    ranking_df_var['combined_rank'] = ranking_df_var[constants.AM_COL_RANK_MAPE] + \
                                      ranking_df_var[constants.AM_COL_RANK_NRMSE_MAXMIN]
    best_row = ranking_df_var.loc[[ranking_df_var['combined_rank'].idxmin()]]
    estimator_type = "linear" if best_row[constants.AM_COL_LINEAR].values[0] else "nonlinear"
    return utils.get_estimator_file_name(estimator_type, best_row[constants.AM_COL_ESTIMATOR].values[0],
                                         target_variable)


def _run_predictions(original_data: pd.DataFrame,
                     input_data: pd.DataFrame,
                     estimators_config: list[Dict[str, Any]],
                     estimator_path: str,
                     output_path: str) -> None:
    """"
    For each target variable in the estimator configuration, load the
    corresponding estimator and run predictions on the input data. Also merges
    all predictions into a single CSV file to make it easier to evaluate
    trade-offs between target variables.
    """
    logger.info("Running predictions")
    target_variables = []

    estimator_folder = utils.get_unpacked_path(estimator_path)

    # run predictions
    for entry in estimators_config:
        target_variable = entry[constants.PRED_CONFIG_TARGET_VAR]
        target_variables.append(target_variable)
        greater_is_better = entry[constants.PRED_CONFIG_GREATER_BETTER]
        if constants.PRED_CONFIG_ESTIMATOR_FILE in entry:
            estimator_file = entry[constants.PRED_CONFIG_ESTIMATOR_FILE]
        else:
            # We consult ranking for each target variable separately, because for some
            # target variables the user mau specify an estimator, while for others not
            # (in which one the best one according to ranking will be taken)
            estimator_file = _get_highest_ranked_estimator(estimator_folder, target_variable)
        logger.info((f"Predicting target variable {target_variable}"
                     f" with {estimator_file}"))
        estimator = utils.load_estimator(
            input_path=estimator_folder, pickle_file=estimator_file)
        y_pred = estimator.predict(input_data)
        predictions_df = input_data.copy(deep=True)
        predictions_df[target_variable] = y_pred

        predictions_ranked_df = _rank_predictions(
            data=predictions_df,
            target_column=target_variable,
            greater_is_better=greater_is_better
        )

        output_file_preds = f"predictions-{target_variable}.csv"
        path = utils.write_df_to_csv(
            df=predictions_ranked_df,
            output_path=output_path,
            output_file=output_file_preds)
        logger.info(f"Wrote predictions to {path}")

    # merge predictions to facilitate demos
    logger.info("Merging predictions")
    all_predictions_files = glob.glob(os.path.join(
        output_path,
        r"predictions-*.csv"))
    dfs = []

    for filename in all_predictions_files:
        # TODO can find better way to handle/avoid this with more time
        ignore_files = [
            constants.PRED_ALL_PREDICTIONS_FILE,
            constants.PRED_GROUND_TRUTH_FILE]
        if any(x in filename for x in ignore_files):
            continue
        df = pd.read_csv(filename, index_col=None, header=0)
        dfs.append(df)
    all_predictions_df = reduce(lambda x, y: pd.merge(x, y), dfs)
    all_predictions_df = all_predictions_df.loc[:, ~all_predictions_df.columns.str.contains('^Unnamed')]
    output_file_all_predictions = constants.PRED_ALL_PREDICTIONS_FILE
    path = utils.write_df_to_csv(
        df=all_predictions_df,
        output_path=output_path,
        output_file=output_file_all_predictions, index=False)
    logger.info(f"Wrote merged predictions to {path}")

    # create comparison with original data

    if original_data is not None and not original_data.empty:
        merged_df = pd.merge(
            original_data,
            all_predictions_df,
            on=input_data.columns.values.tolist(),
            suffixes=["_actual", "_pred"])
        if not merged_df.empty:
            merged_df = merged_df.loc[:, ~merged_df.columns.str.contains('^Unnamed')]
            merged_params = merged_df.columns.values.tolist()
            for p in all_predictions_df:
                if p + "_pred" in merged_params and p + "_actual" in merged_params:
                    merged_df[p + "_mape"] = [mean_absolute_percentage_error([y_t], [y_p]) for y_t, y_p in
                                              zip(merged_df[p + "_actual"], merged_df[p + "_pred"])]
            output_file_merged = constants.PRED_GROUND_TRUTH_FILE
            path = utils.write_df_to_csv(
                df=merged_df,
                output_path=output_path,
                output_file=output_file_merged, index=False)
            logger.info(f"Wrote source of truth to {path}")

            path = utils.write_df_to_csv(
                df=original_data,
                output_path=output_path,
                output_file=constants.PRED_ORIGINAL_TRUTH_FILE, index=False)
            logger.info(f"Wrote original data to {path}")

    if zipfile.is_zipfile(estimator_path):
        # remove temporary estimator artifacts folder created from given archive file
        shutil.rmtree(estimator_folder)


def _rank_predictions(
        data: pd.DataFrame,
        target_column: str,
        greater_is_better: bool = False) -> pd.DataFrame:
    """
    Rank the indicated target variable column according to whether larger 
    or smaller values are preferred. Modifies DataFrame inplace by adding
    suitable ranking column.

    :param data: DataFrame containing predictions.
    :type data: pandas.DataFrame
    :param target_column: Name of the column whose values should be ranked.
        Assumes numeric data.
    :type target_columns: str
    :param greater_is_better: Indicates whether larger or smaller values are
        considered better.
    :type: bool
    :returns: DataFrame with ranking column for each target column.
    """
    data.insert(
        loc=len(data.columns),
        column=f"rank_{target_column}",
        value=data[target_column].rank(ascending=not greater_is_better))
    return data


def demo_predict(original_data: pd.DataFrame, config: PredictionInputSpace,
                 estimator_path: str, feature_engineering: dict[str, str] = None,
                 metadata_parser_class_name: str = None, metadata_path: str = None, output_path: str = None):
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if output_path is None:
        msg = "Must set output_path for demo predict outputs"
        logger.error(msg)
        raise ValueError(msg)

    logger.info("Beginning demo predict")
    _, input_space_df = _create_input_space(
        input_config=config,
        original_data=original_data,
        output_path=output_path,
        feature_engineering=feature_engineering,
        metadata_parser_class_name=metadata_parser_class_name,
        metadata_path=metadata_path
    )
    _run_predictions(
        original_data=original_data,
        input_data=input_space_df,
        estimators_config=config.estimators,
        estimator_path=estimator_path,
        output_path=output_path
    )
    logger.info(f"Demo predict outputs written to {output_path}")

# Predict outputs for every row in a given prediction_data dataframe. If original data ia given, accuracy is computed.
# If delta_only is True, predictions are performed only for the delta between the prediction data and the original data.


def data_predict(original_data: pd.DataFrame, prediction_data: pd.DataFrame, estimator_path: str,
                 estimators_config: list[Dict[str, Any]], target_variables: list[str], delta_only: bool = False,
                 output_path: str = None):
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if output_path is None:
        msg = "Must set output_path for demo predict outputs"
        logger.error(msg)
        raise ValueError(msg)

    prediction_data_inputs = prediction_data[[c for c in prediction_data.columns if c not in target_variables]]
    original_data_inputs = original_data[[c for c in original_data.columns if c not in target_variables]] if \
        original_data is not None and not original_data.empty else None

    #prediction_data_inputs = prediction_data_inputs[prediction_data_inputs.oo.isin([1024])]
    #prediction_data_inputs = prediction_data_inputs[prediction_data_inputs.ii.ge(128)]

    logger.info("Beginning data predict")
    _run_predictions(
        original_data=original_data,
        input_data=prediction_data_inputs[~prediction_data.isin(original_data_inputs)].dropna(how='any') if delta_only
        and original_data_inputs is not None and not original_data_inputs.empty else prediction_data_inputs,
        estimators_config=estimators_config,
        estimator_path=estimator_path,
        output_path=output_path
    )
    logger.info(f"Demo predict outputs written to {output_path}")