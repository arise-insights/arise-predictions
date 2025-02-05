import logging
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass
import yaml
import os
import sys
import importlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV, LeaveOneGroupOut
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingRegressor
from sklearn.inspection import permutation_importance

from arise_predictions.utils import constants, utils
from arise_predictions.metrics import metrics
import shutil

from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)

"""
Performs a parameter search for the models and the search space specified in a
configuration file on the input data (data_execution_history.csv). Persists
best models and evaluates them on held-out test data set. Models (estimators), 
the winning parameter settings, test data, and predictions are all saved as 
output of this script.
In this iteration, we build one model per target variable. 
"""


@dataclass
class EstimatorConfig:

    name: str
    class_name: str
    linear: bool
    parameters: list[Dict[str, Any]]


@dataclass
class EstimatorsConfig:

    estimators: List[EstimatorConfig]
    num_jobs: int


def get_estimators_config(config_file: str, num_jobs: int = -1) -> EstimatorsConfig:
    """
    Read list of learning algorithms and their parameter search space.

    :param config_file: Path to estimator and parameter search space configuration file.
    :type config_file: str
    :param num_jobs: number of processors to use for building the models. Default -1 means all processors are used.
    :type num_jobs: int
    :returns: Data class representation of values from configuration file.
    :rtype: EstimatorsConfig
    """
    logger.info(f"Reading YAML configuration: {config_file}")
    try:
        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f)
    except FileNotFoundError:
        raise ValueError(f"Configuration file not found: {config_file}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing configuration file: {e}")

    return get_estimators_config_from_dict(config_dict=config_dict, num_jobs=num_jobs)


def get_estimators_config_from_dict(config_dict: [str, any], num_jobs: int = -1) -> EstimatorsConfig:
    estimator_configurations = config_dict[constants.AM_CONFIG_ESTIMATORS]
    if not estimator_configurations:
        raise ValueError("No estimator configurations found in given configuration")

    return EstimatorsConfig([EstimatorConfig(entry[constants.AM_CONFIG_NAME], entry[constants.AM_CONFIG_CLASS_NAME],
                                             entry[constants.AM_CONFIG_LINEAR], entry[constants.AM_CONFIG_PARAMETERS])
                             for entry in estimator_configurations], num_jobs)


def _init_estimators(
        config: EstimatorsConfig, cat_indices: list[str]) -> list[tuple[str, Any, str]]:
    """
    Instantiate estimators listed in configuration.

    :param config: Parsed configuration of estimators and parameter search
    space.
    :type config: EstimatorsConfig
    :returns: List of tuples each consisting of descriptive estimator name and
    its instantiation along with whether estimator is linear algorithm.
    :rtype: list[tuple[str, Any, str]]
    """

    # TODO consider changing configuration format to have estimator metadata of
    # which linear is one attribute.

    # TODO consider changing to return dictionary for all estimators

    logger.info("Instantiating estimators from configuration.")
    estimators = []

    for estimator_config in config.estimators:
        name = estimator_config.name
        fqcn = estimator_config.class_name
        linear = estimator_config.linear
        estimator = _instantiate_estimator(fqcn, cat_indices=cat_indices)
        estimators.append((name, estimator, linear))
    
    logger.info(f"Instantiated {len(estimators)} estimators.")
    return estimators


def _instantiate_estimator(fqcn: str, cat_indices: list[str]) -> Any:
    """
    Instantiate estimator from fully-qualified domain name. Fixes random seed.

    :param fqcn: Fully-qualified class name of estimator.
    :type fqcn: str
    :returns: Instantiation of estimator.
    :rtype: Any
    """
    logger.info(f"Instantiating estimator for {fqcn}.")
    module_name, class_name = fqcn.rsplit(".", 1)

    model_module = importlib.import_module(module_name)
    estimator_class = getattr(model_module, class_name)

    # TODO oi weis mir -- there's probably a better way
    if class_name in constants.AM_ESTIMATORS_NO_SEED:
        estimator = estimator_class()
    # TODO oi weis mir -- currently implementing a de-facto explicit initialization for catboost due to
    #  non-standard arguments. E.g., need an explicit silent mode specification since otherwise it inherits the
    #  above noisier log level
    elif class_name in constants.AM_ESTIMATORS_CATBOOST:
        estimator = estimator_class(random_state=constants.SEED,
                                    cat_features=cat_indices,
                                    logging_level='Silent')
    else:
        estimator = estimator_class(random_state=constants.SEED)
    logger.info(f"Instantiated estimator {estimator_class}.")
    return estimator


def _split_data_by_extrapolation_feature(
        data: pd.DataFrame, feature_col: str = None, low_threshold: int = None, 
        high_threshold: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Attempts to split data frame into an interpolation and extrapolation data
    frame by values on the specified feature. The extrapolation data frame
    consists of samples whose feature values are outside the range specified by
    the thresholds. If both thresholds are left at the default value of None,
    this function will return the input data frame.

    :param data: Original data frame from data execution history data.
    :type data: pandas.DataFrame
    :param feature_col: Name of input feature column to split on.
    :type feature_col: str
    :param low_threshold: Exclude samples with feature values less than or
        equal to this.
    :type low_threshold: int
    :param high_threshold: Exclude samples with feature values greater than or
        equal to this.
    :type high_threshold: int
    :returns: Tuple consiting of interpolation and extrapolation data frames. If
        both tresholds are not set, returns only original data frame.
    :rtype: Tuple[pd.DataFrame, pd.DataFrame], where second element can be None.
    """
    # TODO Can probably get rid of several if-statements using bitmasks, but 
    # want to keep things more explicit for now

    # Validate inputs
    if feature_col is None:
        logger.info(("No feature column to extrapolate on has been set." 
                     " Not creating extrapolation data set."))
        return data, None

    if feature_col not in data.columns:
        raise KeyError(("No column with extrapolation feature name:" 
                        f" {feature_col} exists input data. Cannot create" 
                        " extrapolation data set."))

    if low_threshold is not None and data[feature_col].min() > low_threshold:
        msg = (f"No values less than low threshold {low_threshold} exist for" 
               f" extrapolation feature {feature_col}. Proceeding without" 
               " low threshold")
        logger.warning(msg)
        low_threshold = None

    if high_threshold is not None and data[feature_col].max() < high_threshold:
        msg = (f"No values greater than high threshold {high_threshold} exist for" 
               f" extrapolation feature {feature_col}. Proceeding without" 
               " high threshold")
        logger.warning(msg)
        high_threshold = None
    
    if low_threshold is None and high_threshold is None:
        raise ValueError(("No values found for thresholds for feature" 
                     f" {feature_col}. Cannot create extrapolation data set."))
    
    # We are finally ready to split the data set
    if low_threshold is not None and high_threshold is not None:
        interpolation_df = data[(data[feature_col] > low_threshold) & (data[feature_col] < high_threshold)]
        extrapolation_df = data[(data[feature_col] <= low_threshold) | (data[feature_col] >= high_threshold)]
    elif low_threshold is not None:
        interpolation_df = data[data[feature_col] > low_threshold]
        extrapolation_df = data[data[feature_col] <= low_threshold]
    elif high_threshold is not None:
        interpolation_df = data[data[feature_col] < high_threshold]
        extrapolation_df = data[data[feature_col] >= high_threshold]

    return interpolation_df, extrapolation_df


def _search_models(data: pd.DataFrame, estimators: list[tuple[str, Any]], 
                   config: EstimatorsConfig, categorical_variables: list[str],
                   target_variables: List[str], 
                   output_path: str, leave_one_out_cv: str = None,
                   feature_col: str = None, low_threshold: int = None, 
                   high_threshold: int = None, randomized_hpo: bool = False,
                   n_random_iter: int = constants.AM_DEFAULT_N_ITER_RANDOM_HPO) -> Tuple[pd.DataFrame, Dict]:
    """
    Run parameter search for each target variable using the estimators and 
    their parameter search spaces from configuration.

    :param data: Data execution history data preprocessed from measurements.
    :type data: pandas.DataFrame
    :param estimators: List of tuples with estimator names and classes.
    :type estimators: list[tuple[str, Any]].
    :param config: Estimators and parameter search spaces parsed from
    configuration file.
    :type config: EstimatorsConfig
    :param target_variables: List of column names containing target 
    variables (from job_spec.yaml). 
    :type target_variables: List[Tuple[str, Any]]
    :param output_path: Path for storing results.
    :type output_path: str
    :param leave_one_out_cv: feature to split cross validation. None for kfold cross validation.
    :type leave_one_out_cv: str
    :param feature_col: Name of input feature column to split on for extrapolation.
    :type feature_col: str
    :param low_threshold: Exclude samples with feature values less than or
        equal to this.
    :type low_threshold: int
    :param high_threshold: Exclude samples with feature values greater than or
        equal to this.
    :type high_threshold: int
    :param randomized_hpo: Use randomized sampling instead of exhaustive search for HPO
    :type randomized_hpo: bool
    :param n_random_iter: Number of sampling iterations for each model HP space
    :type n_random_iter: int
    :returns: Tuple consisting of data frame of the best parameters per
    estimator for each target variable and dictionary to enable retrieval of
    the best estimator and parameters per target variable upon ranking.
    :rtype: Tuple[pd.DataFrame, Dict]
    """
    logger.info(("Beginning parameter search for target variables"
                 f" {target_variables} using {len(estimators)}."))

    # commenting out since these are already computed outside the function and passed to it as an argument
    #categorical_variables: list[str] = utils.get_categorical_features(data)
    #logger.info(f"Found categorical variables: {categorical_variables}")

    num_jobs = config.num_jobs  # default use all cores
    logger.info(f'Using {constants.AM_NUM_JOBS}={num_jobs}')

    if leave_one_out_cv:
        cv_generator = LeaveOneGroupOut()
    else:
        cv_generator = KFold(n_splits=10, shuffle=True)

    scoring = metrics.create_scorers()
    result_stats = []

    # dict from target_var to estimator names, estimators, their best params
    # target_var_name: [{estimator_name: est1, linear: True, 
    # estimator_class: class_est1, best_parameters: {params}}, ...]
    estimators_per_target_vars = {}

    # is there an intention to create extrapolation data set?
    if feature_col is not None:
        data, extrapolation_data = _split_data_by_extrapolation_feature(
            data=data, feature_col=feature_col, low_threshold=low_threshold,
            high_threshold=high_threshold)
    
        if extrapolation_data is not None:
            path = utils.write_df_to_csv(df=extrapolation_data,
                                         output_path=output_path,
                                         output_file=constants.EXTRA_TEST_FILE)
            logger.info(f"Extrapolation test data written to {path}")

    for target_variable in target_variables:
        logger.info(("Begin parameter search for target variable:"
                     f" {target_variable}"))

        target_var_estimators = []

        y = data[target_variable]
        X = data.drop(target_variables, axis=1)

        # Create held-out test set to run predictions on final model.
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=constants.SEED)
        
        target_var_min, target_var_max = metrics.get_min_max_from_array(y_train)

        if leave_one_out_cv:
            groups = X_train.groupby(leave_one_out_cv.split(',')).ngroup()
        else:
            groups = None

        # persist training data
        train_data = pd.concat([X_train, y_train], axis=1)
        output_file_train_data = f"train-data-{target_variable}.csv"
        utils.write_df_to_csv(train_data, output_path=output_path,
                              output_file=output_file_train_data, index=False)

        # persist test data
        test_data = pd.concat([X_test, y_test], axis=1)
        output_file_test_data = f"test-data-{target_variable}.csv"
        utils.write_df_to_csv(test_data, output_path=output_path,
                              output_file=output_file_test_data, index=False)
        logger.info(("Test data written to" 
                     f" {os.path.join(output_path, output_file_test_data)}"))
        
        logger.info("Setting up hyperparameter search")

        for estimator_name, estimator_class, linear in estimators:
            for estimator in config.estimators:
                if estimator.name == estimator_name:
                    params = estimator.parameters

                    preprocessor = ColumnTransformer(
                        transformers=[
                            ("cat", OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                                feature_name_combiner="concat"),
                                [] if estimator_class.__class__.__name__ in
                                constants.AM_ESTIMATORS_CATBOOST else categorical_variables)
                        ],
                        remainder="passthrough")

                    pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                                               ("estimator", TransformedTargetRegressor(regressor=estimator_class,
                                                                                        func=np.log1p,
                                                                                        inverse_func=np.expm1))])

                    if randomized_hpo:
                        search = RandomizedSearchCV(
                            estimator=pipeline,
                            param_distributions=params,
                            scoring=scoring,
                            n_jobs=num_jobs,
                            n_iter=n_random_iter if n_random_iter is not None else
                            constants.AM_DEFAULT_N_ITER_RANDOM_HPO,
                            refit=constants.AM_DEFAULT_METRIC,
                            cv=cv_generator,
                            verbose=1)
                    else:
                        search = GridSearchCV(
                            estimator=pipeline,
                            param_grid=params,
                            scoring=scoring,
                            n_jobs=num_jobs,
                            refit=constants.AM_DEFAULT_METRIC,
                            cv=cv_generator,
                            verbose=1)

                    logger.info(("Commencing parameter search on train set"
                                 f" for target variable: {target_variable}"
                                 " with this estimator and parameters:" 
                                 f" {estimator_name}"))

                    if groups is not None:
                        search.fit(X_train, y_train, groups=groups)
                    else:
                        search.fit(X_train, y_train)

                    logger.info(("Search results for target variable" 
                                 f" {target_variable}" 
                                 f" with estimator {estimator_name}:"))
                    logger.info(f"Best score (MAPE): {search.best_score_:.3f}")
                    logger.info(f"Best params: {search.best_params_}")
                    logger.info(f"Best estimator: {search.best_estimator_}")

                    # persist cv results
                    cv_results_df = pd.DataFrame(search.cv_results_)
                    output_file_cv_results = (f"cv_results-{estimator_name}"
                                              f"-{target_variable}.csv")
                    path = utils.write_df_to_csv(
                        df=cv_results_df, 
                        output_path=output_path,
                        output_file=output_file_cv_results)
                    logger.info(f"Wrote parameter search results to {path}")

                    # This is the best result per target variable for each
                    # estimator. In other words, the best version 
                    # of each estimator for a given target variable. 
                    # That is what goes into result_stats and thereby
                    # into stats_df. target_var_estimators also stores the best
                    # version/hyper-parameters for each estimator per
                    # target variable.
                    # We use this data structure to select the best estimators
                    # later on.

                    rank_col = f"rank_test_{constants.AM_DEFAULT_METRIC}"
                    best_cv_row = cv_results_df[
                        cv_results_df[rank_col] == 1].nsmallest(
                            1, 'mean_fit_time') 
                    best_cv_row.insert(
                        0, constants.AM_COL_ESTIMATOR, estimator_name)
                    best_cv_row.insert(
                        1, constants.AM_COL_LINEAR, linear)
                    best_cv_row.insert(
                        2, constants.AM_COL_TARGET, target_variable)
                    best_cv_row.insert(
                        3, 
                        constants.AM_COL_TARGET_RANGE, 
                        f"[{target_var_min:.3f}, {target_var_max:.3f}]")
                    result_stats.append(best_cv_row) 

                    # For each target variable, record the best version of each
                    # estimator along with some metadata about it and the
                    # best hyperparameters for this target_var-estimator pair.
                    target_var_estimators.append({
                        "estimator_name": estimator_name, 
                        "linear": linear,
                        "estimator_class": search.best_estimator_, 
                        "best_parameters": search.best_params_})
                    estimators_per_target_vars[
                        target_variable] = target_var_estimators

    stats_df = pd.DataFrame(pd.concat(result_stats, ignore_index=True))

    # remove columns not needed in rankings and stored in cv results
    stats_df = stats_df[stats_df.columns.drop(list(stats_df.filter(
        regex="^split")))]
    stats_df = stats_df[stats_df.columns.drop(list(stats_df.filter(
        regex="^rank_test")))]
    stats_df = stats_df[stats_df.columns.drop(list(stats_df.filter(
        regex="^param_estimator")))]

    # move params columns to end
    column_to_move = stats_df.pop("params")
    stats_df.insert(len(stats_df.columns), "params", column_to_move)

    return stats_df, estimators_per_target_vars


def _rank_estimators(summary_stats: pd.DataFrame,
                     output_path: str,
                     output_file: str) -> pd.DataFrame:
    """
    Rank estimators per target variable for several performance metrics.

    :param summary_stats: Data frame of the best-performing parameters per
        estimator for each target variable along with key performance metrics.
    :type: data: pandas.DataFrame
    :param output_path: Path for storing results.
    :type output_path: str 
    :param output_file: File name.
    :type output_file: str
    :returns: Data frame with ranking of estimators per target variable along
        several performance metrics (a subset we understand, will ad more in
        future). 
    :rtype: pd.DataFrame
    """
    logger.info("Ranking estimators for all target variables")
    summary_stats.insert(
        3, 
        constants.AM_COL_RANK_MAPE, 
        summary_stats.groupby(
            [constants.AM_COL_TARGET])[
                constants.AM_METRIC_MAPE_MEAN].rank(ascending=False))
    summary_stats.insert(
        4, 
        constants.AM_COL_RANK_NRMSE_MAXMIN, 
        summary_stats.groupby(
            [constants.AM_COL_TARGET])[
                constants.AM_METRIC_NRMSE_MEAN].rank(ascending=False))
    summary_stats.insert(
        5, 
        constants.AM_COL_RANK_R2, 
        summary_stats.groupby(
            [constants.AM_COL_TARGET])[
                constants.AM_METRIC_R2_MEAN].rank(ascending=False))

    output_file_rankings = output_file
    path = utils.write_df_to_csv(
        df=summary_stats, 
        output_path=output_path, 
        output_file=output_file_rankings)
    logger.info(f"Wrote estimator rankings to {path}")

    return summary_stats


def _select_best_estimators(
        target_variables: list[str],
        rankings: pd.DataFrame, 
        estimators_per_target_variable: Dict[str, Dict[str, Any]],
        output_path: str,
        num_jobs: int,
        leave_one_out_cv: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Determine the best estimator per target variable from the ranking of
    estimators (in this iteration using the default performance metric MAPE),
    persist estimator, its best parameter settings, and run predictions on
    held-out, unseen data.

    :returns: Dictionary from target variable to list of best estimators for 
    this target variable. Example:
    'gpu': [{'estimator_name': 'a', 'linear': True, 'estimator_class': Pipeline,
    'best_parameters': {'gamma': 0.5}}]
    :rtype: Dict[str, List[Dict[str, Any]]]
    """
    logger.info(f"Selecting best estimator per target variable for target variables: {target_variables}")

    # collect results
    test_result_rows = []
    extrapolation_test_results_rows = []
    best_estimators_for_target_variables = {}

    for target_var in target_variables:
        best_estimators = []
        best_linear_estimator_for_target_var = rankings[
            (rankings[constants.AM_COL_TARGET] == target_var) &
            (rankings[constants.AM_COL_LINEAR] == True)].loc[
            lambda filtered: filtered[constants.AM_COL_RANK_MAPE] == filtered[constants.AM_COL_RANK_MAPE].min()][
            constants.AM_COL_ESTIMATOR].values[0]
        logger.info((f"Best linear estimator for target variable {target_var}" 
                     f" is {best_linear_estimator_for_target_var}"))
        
        best_nonlinear_estimator_for_target_var = rankings[
            (rankings[constants.AM_COL_TARGET] == target_var) &
            (rankings[constants.AM_COL_LINEAR] == False)].loc[
            lambda filtered: filtered[constants.AM_COL_RANK_MAPE] == filtered[constants.AM_COL_RANK_MAPE].min()][
            constants.AM_COL_ESTIMATOR].values[0]
        logger.info((f"Best non-linear estimator for target variable {target_var}" 
                     f" is {best_nonlinear_estimator_for_target_var}"))

        estimators = estimators_per_target_variable[target_var]
        best_estimators_for_target_variables[target_var] = best_estimators

        for estimator in estimators:
            if estimator["estimator_name"] == best_linear_estimator_for_target_var or \
                    estimator["estimator_name"] == best_nonlinear_estimator_for_target_var:
                best_estimators.append(estimator)
            
                test_performance_summary, extrapolation_performance_summary = _test_and_persist_estimator(
                    estimator=estimator,
                    target_variables=target_variables,
                    target_variable=target_var,
                    output_path=output_path,
                    num_jobs=num_jobs,
                    leave_one_out_cv=leave_one_out_cv)
                test_result_rows.append(test_performance_summary)
                extrapolation_test_results_rows.append(extrapolation_performance_summary)

        # persist test performance summary
        performance_summary_df = pd.DataFrame(test_result_rows)
        output_file_performance_summary = ("best-estimators-testset-"
                                           "performance-summary.csv")
        utils.write_df_to_csv(
            df=performance_summary_df, 
            output_path=output_path, 
            output_file=output_file_performance_summary)
        
        extrapolation_summary_df = pd.DataFrame(
            extrapolation_test_results_rows)
        output_file_extrapolation_performance_summary = (
            "best-estimators-extrapolation-testset-performance-summary.csv")
        
        utils.write_df_to_csv(
            df=extrapolation_summary_df,
            output_path=output_path,
            output_file=output_file_extrapolation_performance_summary)

    return best_estimators_for_target_variables


def _test_and_persist_estimator(
        estimator: Dict[str, Any],
        target_variables: List[str], 
        target_variable: str,
        output_path: str,
        num_jobs: int,
        leave_one_out_cv: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Performs tests to measure a given estimator performance. Then train on all data
    (CV + test) and persists it.

    :param estimator: Dictionary with information about the estimator
    :type estimator: Dict[str, Any]
    :param target_variables: List of target variables
    :type target_variables: List[str]
    :param target_variable: Name of the target variable on which the estimator
        should be tested
    :type target_variable: str
    :param output_path: Path for storing results.
    :type output_path: str 
    :returns: Dictionaries with performance summary for test sets.
    :rtype: Tuple[Dict[str], Dict[str]] 
    """

    # Test on test set (samples held out during training)
    test_file = f"test-data-{target_variable}.csv"
    output_file_predictions = (f"predictions-"
                               f"{estimator['estimator_name']}-"
                               f"{target_variable}.csv")
    test_performance_summary = _predict_on_test(
        estimator_name=estimator["estimator_name"],
        estimator_class=estimator["estimator_class"],
        target_variables=target_variables,
        target_variable=target_variable,
        test_file=test_file,
        output_path=output_path,
        output_file_predictions=output_file_predictions)
    
    if not test_performance_summary:
        logger.warning(("Test results for estimator:"
                        f" {estimator['estimator_name']} on"
                        f" target variable: {target_variable}"
                        f" using test data: {test_file}"
                        " is empty"))
        
    # Test on extrapolation test set (unseen values for features)
    # If no extrapolation test set exists, logs warning and moves on
    output_file_predictions = ("extrapolation-predictions-"
                               f"{estimator['estimator_name']}"
                               f"-{target_variable}.csv")
    extrapolation_performance_summary = _predict_on_test(
        estimator_name=estimator["estimator_name"],
        estimator_class=estimator["estimator_class"],
        target_variables=target_variables,
        target_variable=target_variable,
        test_file=constants.EXTRA_TEST_FILE,
        output_path=output_path,
        output_file_predictions=output_file_predictions)
        
    if not extrapolation_performance_summary:
        logger.warning(("Extrapolation test results for estimator:"
                        f" {estimator['estimator_name']} on"
                        f" target variable: {target_variable}"
                        f" using test data: {test_file}"
                        " is empty"))

    _fit_on_all_data(estimator=estimator, target_variable=target_variable, target_variables=target_variables,
                     output_path=output_path, num_jobs=num_jobs,
                     leave_one_out_cv=leave_one_out_cv)

    _persist_estimator(
        estimator_name=estimator["estimator_name"],
        estimator_class=estimator["estimator_class"],
        estimator_linear=estimator["linear"],
        best_params=estimator["best_parameters"],
        target_variable=target_variable,
        output_path=output_path
    )
    
    return test_performance_summary, extrapolation_performance_summary


def _fit_on_all_data(estimator: Any, target_variable: str, target_variables: list[str], output_path: str,
                     num_jobs: int, leave_one_out_cv: str):

    train_data = _get_data(f"train-data-{target_variable}.csv", output_path)
    test_data = _get_data(f"test-data-{target_variable}.csv", output_path)

    data = pd.concat([train_data, test_data], ignore_index=True)

    y_train = data[target_variable]
    X_train = data.drop(target_variables, axis=1, errors="ignore")

    estimator_class = estimator["estimator_class"]

    pipeline = Pipeline(steps=[("estimator", TransformedTargetRegressor(regressor=estimator_class,
                                                                                        func=np.log1p,
                                                                                        inverse_func=np.expm1))])

    if leave_one_out_cv:
        groups = X_train.groupby(leave_one_out_cv.split(',')).ngroup()
        cv_generator = LeaveOneGroupOut()
    else:
        groups = None
        cv_generator = KFold(n_splits=10, shuffle=True)

    search = GridSearchCV(
            estimator=pipeline,
            param_grid={'estimator__regressor__'+key: [value] for key, value in estimator["best_parameters"].items()},
            scoring=metrics.create_scorers(),
            n_jobs=num_jobs,
            refit=constants.AM_DEFAULT_METRIC,
            cv=cv_generator,
            verbose=1)

    if groups is not None:
        search.fit(X_train, y_train, groups=groups)
    else:
        search.fit(X_train, y_train)


def _get_data(input_file_name: str, output_path: str) -> Any:
    input_file = os.path.join(output_path, input_file_name)

    if not os.path.exists(input_file):
        raise ValueError(f"No file {input_file} found for data set")

    return pd.read_csv(input_file)


def _persist_estimator(estimator_name: str, estimator_class: Any, 
                       estimator_linear: bool,
                       best_params: Dict[str, str], target_variable: str, 
                       output_path: str) -> None:
    """
    Persist an estimator and its best hyperparameters. In our case, the best
    estimator is in the form of a Pipeline.
    """
    estimator_type = "linear" if estimator_linear else "nonlinear"

    logger.info(("Persisting estimator and best parameters for" 
                 f" {estimator_type} estimator {estimator_name} and" 
                 f" target variable {target_variable}"))

    # persist best estimator
    path = utils.persist_estimator(
        estimator=estimator_class, 
        output_path=output_path,
        output_file=utils.get_estimator_file_name(estimator_type, estimator_name, target_variable))
    logger.info(f"Persisted estimator to {path}")

    # store information about best params
    output_file_params = ((f"params-{estimator_type}-{estimator_name}"
                           f"-{target_variable}.yaml"))
    with open(os.path.join(output_path, output_file_params), "w") as f:
        yaml.dump(best_params, f, default_flow_style=False)
    logger.info((f"Persisted best parameters to" 
                 f" {os.path.join(output_path, output_file_params)}"))


def _predict_on_test(estimator_name: str, estimator_class: Any, 
                     target_variables: List[str], target_variable: str, 
                     test_file: str, output_path: str,
                     output_file_predictions: str) -> Dict[str, str]:
    """
    Run predict for the given estimator for the given target variable on the 
    given test set to measure estimator performance. Can be used to predict on
    unseen (held-out during training) test to gain an additionl performance
    estimate or to test ability to extrapolate  on feature input values that 
    were not seen.
    """
    logger.info((f"Predicting on test data for target variable:" 
                 f" {target_variable} using estimator: {estimator_name}"))

    input_file = os.path.join(output_path, test_file)

    if not os.path.exists(input_file):
        logger.warning((f"No file {input_file} found for test data set."
                        "Skipping predictions."))
        return {}
        
    test_data = pd.read_csv(input_file)
    y_test = test_data[target_variable]
    X_test = test_data.drop(target_variables, axis=1, errors="ignore")

    y_pred = estimator_class.predict(X_test)

    # compute performance metrics
    mape_mean_test, mape_std_test, nrmse_mean_test, r2_mean_test = metrics.compute_test_metrics(y_true=y_test, y_pred=y_pred)

    logger.info((f"Performance of estimator: {estimator_name}"
                 f" for target variable: {target_variable}"
                 f" on test data: {input_file}:"))
    logger.info(f"MAPE mean (extrapolation test): {mape_mean_test:.3f}")
    logger.info(f"NRMSE (minmax) mean (extrapolation test): {nrmse_mean_test:.3f}")
    logger.info(f"R2 mean (extrapolation test): {r2_mean_test:.3f}")

    # compute feature importance
    current_backend = plt.get_backend()
    plt.switch_backend('Agg') # to avoid showing the bar chart
    perm_importance = permutation_importance(estimator_class, X_test, y_test, n_repeats=10, random_state=0)
    sorted_idx = perm_importance.importances_mean.argsort()
    plt.barh(X_test.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
    plt.xlabel(f"Permutation Importance for {target_variable}")
    plt.savefig(os.path.join(output_path, f"perm-importance-{estimator_name}-{target_variable}.png"))
    plt.switch_backend(current_backend)

    # persist predictions
    df_X_test = pd.DataFrame(data=X_test)
    df_y_test = pd.DataFrame(data=y_test, columns=[f"{target_variable}"])
    df_predictions = pd.DataFrame(data=y_pred, columns=["prediction"])
    df_results = df_X_test.reset_index()
    df_results[f"{target_variable}"] = df_y_test.reset_index()[
        f"{target_variable}"]
    df_results["prediction"] = df_predictions.reset_index()["prediction"]

    utils.write_df_to_csv(
        df=df_results, 
        output_path=output_path, 
        output_file=output_file_predictions)

    # return performance metrics
    performance = {constants.AM_COL_ESTIMATOR: estimator_name,
                   constants.AM_COL_TARGET: target_variable,
                   "MAPE_mean_test": mape_mean_test,
                   "MAPE_std_test": mape_std_test,
                   "NRMSE(maxmin)_mean_test": nrmse_mean_test,
                   "R2_mean_test": r2_mean_test}
    return performance


def _search_meta_model(
        best_estimators_per_target_var: Dict[str, List[Dict[str, Any]]],
        config: EstimatorsConfig,
        cat_indices: List[str],
        target_variables: List[str],
        output_path: str,
        leave_one_out_cv: str = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run parameter search to build meta-learner for extrapolation. One per 
    target variable. 

    :param best_estimators_per_target_var: Dictionary from target variable to
        list of best linear and non-linear estimators (currently one each). This
        gets returned by _select_best_estimators().
    :type best_estimators_per_target_var: Dict[str, List[Dict[str, Any]]] 
    """
    logger.info(("Beginning meta-learner parameter search for target"
                 f" variables: {target_variables}"))

    num_jobs = config.num_jobs
    logger.info(f'Using {constants.AM_NUM_JOBS}={num_jobs}')

    if leave_one_out_cv:
        cv_generator = LeaveOneGroupOut()
    else:
        cv_generator = KFold(n_splits=10, shuffle=True)

    scoring = metrics.create_scorers()
    result_stats = []
    meta_learner_per_target_var = {}

    for target_variable in target_variables:
        logger.info(("Begin meta-learner parameter search for target variable:"
                     f" {target_variable}"))
        
        target_var_estimators = []

        train_file = os.path.join(output_path, f"test-data-{target_variable}.csv")

        if not os.path.exists(train_file):
            logger.warning((f"No file {train_file} found for test data set."
                            " Skipping meta-learner search for" 
                            f" target variable: {target_variable}."))
            continue

        data = pd.read_csv(train_file)
        y_train = data[target_variable]
        X_train = data.drop(target_variable, axis=1)

        target_var_min, target_var_max = metrics.get_min_max_from_array(y_train)
        
        if leave_one_out_cv:
            groups = X_train.groupby(leave_one_out_cv.split(',')).ngroup()
        else:
            groups = None
        
        logger.info("Setting up meta-learner hyperparameter search")
        meta_learner_name = "ElasticNet-Regression"
        meta_estimator = next((
            est for est in config.estimators
            if est.name == meta_learner_name), None)

        if not meta_estimator:
            logger.warning((f"Could not find estimator: {meta_learner_name}"
                            " in model search configuration. Skipping"
                            " meta-learner search for target variable:" 
                            f" {target_variable}"))
            continue

        meta_learner_fqcn = meta_estimator.class_name
        meta_learner_params = meta_estimator.parameters
        meta_learner_linear = meta_estimator.linear
        meta_learner_class = _instantiate_estimator(
            fqcn=meta_learner_fqcn, cat_indices=cat_indices)
        estimators = best_estimators_per_target_var[target_variable]
        component_estimators = [
            (estimator["estimator_name"], 
             estimator["estimator_class"])
            for estimator in estimators]
        meta_learner = StackingRegressor(
            estimators=component_estimators, 
            final_estimator=meta_learner_class)

        # Limit hyperparameter search to the final estimator
        meta_learner_params = {"final_" + key: value for key, value in meta_learner_params.items()}
        search = GridSearchCV(
            estimator=meta_learner, 
            param_grid=meta_learner_params,
            scoring=scoring,
            n_jobs=num_jobs,
            refit=constants.AM_DEFAULT_METRIC,
            cv=cv_generator,
            verbose=1)
        
        logger.info(("Commencing meta-learner parameter search on train set"
                     f" for target variable: {target_variable}"
                     " with this estimator and parameters:" 
                     f" {meta_learner_name}"))
        
        if groups is not None:
            search.fit(X_train, y_train, groups=groups)
        else:
            search.fit(X_train, y_train)

        logger.info(("Search results for target variable" 
                        f" {target_variable}" 
                        f" with meta-learner estimator {meta_learner_name}:"))
        logger.info(f"Best score (MAPE): {search.best_score_:.3f}")
        logger.info(f"Best params: {search.best_params_}")
        logger.info(f"Best estimator: {search.best_estimator_}")

        # persist cv results
        cv_results_df = pd.DataFrame(search.cv_results_)
        output_file_cv_results = (f"cv_results-meta-learner-{meta_learner_name}"
                                    f"-{target_variable}.csv")
        path = utils.write_df_to_csv(
            df=cv_results_df, 
            output_path=output_path,
            output_file=output_file_cv_results)
        logger.info(f"Wrote parameter search results to {path}")
    
        # Keep track of best version of meta-learner per target variable in
        # a way that allows us to add to all-star-rankings later.
        rank_col = f"rank_test_{constants.AM_DEFAULT_METRIC}"
        best_cv_row = cv_results_df[
            cv_results_df[rank_col] == 1].nsmallest(
                1, 'mean_fit_time') 
        best_cv_row.insert(
            0, constants.AM_COL_ESTIMATOR, constants.AM_META_LEARNER_PREFIX + meta_learner_name)
        best_cv_row.insert(
            1, constants.AM_COL_LINEAR, meta_learner_linear)
        best_cv_row.insert(
            2, constants.AM_COL_TARGET, target_variable)
        best_cv_row.insert(
            3, 
            constants.AM_COL_TARGET_RANGE, 
            f"[{target_var_min:.3f}, {target_var_max:.3f}]")
        result_stats.append(best_cv_row) 
    
        # Keep track of best meta-learner version per target variable
        target_var_estimators.append({
            "estimator_name": constants.AM_META_LEARNER_PREFIX + meta_learner_name,
            "linear": meta_learner_linear,
            "estimator_class": search.best_estimator_,
            "best_parameters": search.best_params_
        })
        meta_learner_per_target_var[target_variable] = target_var_estimators
    
    stats_df = pd.DataFrame(pd.concat(result_stats, ignore_index=True))

    # remove columns not needed in rankings and stored in cv results
    stats_df = stats_df[stats_df.columns.drop(list(stats_df.filter(
        regex="^split")))]
    stats_df = stats_df[stats_df.columns.drop(list(stats_df.filter(
        regex="^rank_test")))]
    stats_df = stats_df[stats_df.columns.drop(list(stats_df.filter(
        regex="^param_final_estimator")))]

    # move params columns to end
    column_to_move = stats_df.pop("params")
    stats_df.insert(len(stats_df.columns), "params", column_to_move)

    return stats_df, meta_learner_per_target_var


def _merge_rankings(output_path: str):
    """"
    Merge meta-learner results into all star rankings if there is a meta-learner
    rankings file.
    """
    meta_rankings_file = os.path.join(
        output_path,
        constants.AM_META_LEARNER_PREFIX + constants.AM_RANKINGS_FILE
    )
    all_rankings_file = os.path.join(
        output_path,
        constants.AM_RANKINGS_FILE
    )

    if not os.path.exists(meta_rankings_file):
        logger.warning(f"No meta-learner rankings file : {meta_rankings_file} found.")
        return None
    elif not os.path.exists(all_rankings_file):
        logger.warning(f"No rankings file : {all_rankings_file} found.")
        return None

    meta_rankings_df = pd.read_csv(meta_rankings_file)
    all_rankings_df = pd.read_csv(all_rankings_file)
    combined_rankings_df = pd.concat([meta_rankings_df, all_rankings_df], ignore_index=True)
    combined_rankings_df[constants.AM_COL_RANK_MAPE] = (
        combined_rankings_df.groupby(constants.AM_COL_TARGET)[
            constants.AM_METRIC_MAPE_MEAN].rank(ascending=False)
        )
    combined_rankings_df[constants.AM_COL_RANK_NRMSE_MAXMIN] = (
        combined_rankings_df.groupby(constants.AM_COL_TARGET)[
            constants.AM_METRIC_NRMSE_MEAN].rank(ascending=False)
        )
    combined_rankings_df[constants.AM_COL_RANK_R2] = (
        combined_rankings_df.groupby(constants.AM_COL_TARGET)[
            constants.AM_METRIC_R2_MEAN].rank(ascending=False)
        )

    path = utils.write_df_to_csv(
        df=combined_rankings_df,
        output_path=output_path,
        output_file=constants.AM_RANKINGS_FILE
    )
    logger.info(f"Write combined (regular and meta-learner) estimator rankings to {path}")


def _persist_and_test_meta_estimator(
        meta_learner_per_target_variable: Dict[str, Any],
        output_path: str,
        num_jobs: int,
        leave_one_out_cv: str) -> None:
    """
    Persist the meta-estimator for each target variable and evaluate it against
    corresponding test sets and extrapolation test set.
    """
    logger.info("Persist and test meta learner")

    test_result_rows = []
    extrapolation_test_result_rows = []

    target_variables = meta_learner_per_target_variable.keys()

    for target_var, meta_estimators in meta_learner_per_target_variable.items():
        meta_estimator = meta_estimators[0]  # Only 1 in list for meta-learner
        test_performance_summary, extrapolation_performance_summary = _test_and_persist_estimator(
            estimator=meta_estimator,
            target_variables=target_variables,
            target_variable=target_var,
            output_path=output_path,
            num_jobs=num_jobs,
            leave_one_out_cv=leave_one_out_cv)
        test_result_rows.append(test_performance_summary)
        extrapolation_test_result_rows.append(extrapolation_performance_summary)

    # persist test performance summary
    performance_summary_df = pd.DataFrame(test_result_rows)
    output_file_performance_summary = ("best-meta-estimators-testset-"
                                       "performance-summary.csv")
    utils.write_df_to_csv(
        df=performance_summary_df, 
        output_path=output_path, 
        output_file=output_file_performance_summary)
    
    extrapolation_summary_df = pd.DataFrame(
        extrapolation_test_result_rows)
    output_file_extrapolation_performance_summary = (
        "best-meta-estimators-extrapolation-testset-performance-summary.csv")
    
    utils.write_df_to_csv(
        df=extrapolation_summary_df,
        output_path=output_path,
        output_file=output_file_extrapolation_performance_summary)


def auto_build_models(raw_data: pd.DataFrame, config: EstimatorsConfig,
                      target_variables: list[str], output_path: str = None, 
                      leave_one_out_cv: str = None, feature_col: str = None,
                      low_threshold: int = None, high_threshold: int = None,
                      single_output_file: bool = False, randomized_hpo: bool = False,
                      n_random_iter: int = constants.AM_DEFAULT_N_ITER_RANDOM_HPO):
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if output_path is None:
        msg = "Must set output_path for auto build models artifacts"
        logger.error(msg)
        raise ValueError(msg)
    
    logger.info("Beginning auto-model search and build")

    categorical_variables: list[str] = utils.get_categorical_features(raw_data)
    logger.info(f"Found categorical variables: {categorical_variables}")

    categorical_variables_indices = [raw_data.columns.get_loc(var) for var in categorical_variables]

    raw_estimators = _init_estimators(config, cat_indices=categorical_variables_indices)

    stats_df, estimators_per_target_variable = _search_models(
        data=raw_data, 
        estimators=raw_estimators, 
        config=config, 
        categorical_variables=categorical_variables, 
        target_variables=target_variables,
        output_path=output_path, 
        leave_one_out_cv=leave_one_out_cv,
        feature_col=feature_col,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        randomized_hpo=randomized_hpo,
        n_random_iter=n_random_iter)

    rankings_df = _rank_estimators(summary_stats=stats_df,
                                   output_path=output_path,
                                   output_file=constants.AM_RANKINGS_FILE)

    best_estimators_per_target_var = _select_best_estimators(
        target_variables=target_variables, 
        rankings=rankings_df, 
        estimators_per_target_variable=estimators_per_target_variable,
        output_path=output_path,
        num_jobs=config.num_jobs,
        leave_one_out_cv=leave_one_out_cv)
    
    if feature_col:
        logger.info(("Beginning auto-model meta-learner search and build"
                     f" for feature {feature_col}"))
        meta_stats_df, best_meta_learner_per_target_variable = _search_meta_model(
            best_estimators_per_target_var=best_estimators_per_target_var,
            config=config,
            cat_indices=categorical_variables_indices,
            target_variables=target_variables,
            output_path=output_path
        )
        _ = _rank_estimators(
            summary_stats=meta_stats_df,
            output_path=output_path,
            output_file=constants.AM_META_LEARNER_PREFIX + constants.AM_RANKINGS_FILE
        )
        _merge_rankings(output_path=output_path)
        _persist_and_test_meta_estimator(
            meta_learner_per_target_variable=best_meta_learner_per_target_variable,
            output_path=output_path,
            num_jobs=config.num_jobs,
            leave_one_out_cv=leave_one_out_cv
        )
        logger.info(f"Auto-model meta-learner artifacts written to {output_path}")

    if single_output_file:
        archived_output = shutil.make_archive(os.path.join(os.path.dirname(output_path),
                                                           constants.AM_OUTPUT_PATH_SUFFIX), 'zip', output_path)
        if os.path.basename(output_path) == constants.AM_OUTPUT_PATH_SUFFIX:
            # delete output file only if ARISE created it
            shutil.rmtree(output_path)
        logger.info(f"Auto-model artifacts written to {archived_output}")
    else:
        logger.info(f"Auto-model artifacts written to {output_path}")


