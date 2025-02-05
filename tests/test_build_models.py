import unittest
import logging
import sys
import os
import joblib

import pandas as pd

from arise_predictions.utils import utils, constants
from tests.utils.logger_redirector import LoggerRedirector
from arise_predictions.auto_model.build_models import (get_estimators_config,
                                           _init_estimators,
                                           _search_models,
                                           _rank_estimators,
                                           _select_best_estimators)

from arise_predictions.cmd.cmd import parse_args

"""
Tests for auto model building functionality.

We use an auto-model configuration with two models for two target variables for
this test. Each model comes out as the winner. However, I wonder whether it is
possible to get different results once every so many executions that flip the
results or lead to one model being the winner for both target variables. Have
not encountered this, we are setting the seed for models that allow this and
when splitting the data. So probably unlikely, but something to keep in mind.

Also note that the tearDown method deletes any temporary test files that may be
useful for debugging. Have left shutil.rmtree commented out to avoid this.
"""

logger = logging.getLogger('test-logger')

logging.basicConfig(
    format="%(asctime)s %(module)s %(levelname)s: %(message)s",
    level=logging.INFO,
    stream=sys.stdout)


class TestBuildModels(unittest.TestCase):

    def setUp(self):
        # unittest has reassigned sys.stdout and sys.stderr by this point
        LoggerRedirector.redirect_loggers(fake_stdout=sys.stdout, 
                                          fake_stderr=sys.stderr) 
        
        # Simulate command-line arguments
        parse_args(["--num-jobs", "-1"])
        
        self.resources_path = "tests/resources"
        self.config_file = os.path.join(self.resources_path, "model-search-config.yaml")
        self.data_file = "mlcommons_inference_data_execution_history.csv"
        self.target_variables = ["tokens_per_second"]

        data = pd.read_csv(os.path.join(self.resources_path, self.data_file))
        self.categorical_variables: list[str] = utils.get_categorical_features(data)
        self.categorical_variables_indices = [data.columns.get_loc(var) for var in self.categorical_variables]

        self.script_dir = os.path.dirname(__file__)
        self.output_path = os.path.abspath(os.path.join(
               self.script_dir, "test-tmp/ARISE-auto-models"))

        # self.best_model_gpu = "model-Ridge-gpu_memory_utilization_max.pkl"
        # self.best_model_time = "model-LinearRegression-train_runtime.pkl"
        # self.best_params_gpu = "params-Ridge-gpu_memory_utilization_max.yaml"
        # self.best_params_time = "params-LinearRegression-train_runtime.yaml"
        self.train_data = "train-data-tokens_per_second.csv"

        self.results_ridge = ("cv_results-RidgeRegression-tokens_per_second"
                                  ".csv")
        self.results_lr = "cv_results-LinearRegression-tokens_per_second.csv"
        self.results_xgb = "cv_results-XGBoost-Regressor-tokens_per_second.csv"
        
        self.rankings = "all-star-rankings.csv"

        self.estimator_linear = "estimator-linear-RidgeRegression-tokens_per_second.pkl"
        self.estimator_nonlinear = "estimator-nonlinear-XGBoost-Regressor-tokens_per_second.pkl"

        self.predictions_lr = "predictions-LinearRegression-tokens_per_second.csv"
        self.predictions_ridge = "predictions-RidgeRegression-tokens_per_second.csv"
        self.predictions_xgb = "predictions-XGBoost-Regressor-tokens_per_second.csv"

        self.best_estimators_summary = "best-estimators-testset-performance-summary.csv"

        self.feature_col = "# of Accelerators"
        self.high_threshold = 6
        self.extrapolation_file = "extrapolation-test-data.csv"

    def tearDown(self):
        LoggerRedirector.reset_loggers(
            fake_stdout=sys.stdout, fake_stderr=sys.stderr)
        
        # if os.path.exists(self.output_path):
        #     shutil.rmtree(self.output_path)

    def test_reads_estimators_and_params_from_config(self):
        """
        Reads estimators and their hyperparameters from the configuration file.
        """
        config = get_estimators_config(self.config_file)
        estimators = config.estimators
        logger.error(f"estimators from YAML:\n{estimators}")
        self.assertEqual(4, len(estimators))
        self.assertTrue(any(d.name == "RidgeRegression" for d in estimators))
    
    def test_setup_estimators(self):
        """
        List of estimators is instantiated according to configuration.
        """
        config = get_estimators_config(self.config_file)
        estimators = _init_estimators(
            config, self.categorical_variables_indices)
        self.assertEqual(len(estimators), 
                         len(config.estimators))
        estimator_names = [x[0] for x in estimators]
        config_names = [x.name for x in config.estimators]
        logger.error(f"estimator names: {estimator_names}")
        logger.error(f"config names: {config_names}")
        self.assertTrue(set(estimator_names) == set(config_names))
    
    def test_training_files_exist(self):
        """
        CSV files containing training data have been written.
        """
        config = get_estimators_config(self.config_file)
        estimators = _init_estimators(
            config, self.categorical_variables_indices)
        data = pd.read_csv(os.path.join(self.resources_path, self.data_file))
        _search_models(data=data, estimators=estimators, config=config,
                       categorical_variables=self.categorical_variables,
                       target_variables=self.target_variables, 
                       output_path=self.output_path)
        self.assertTrue(os.path.exists(os.path.join(
            self.output_path,
            self.train_data
        )))

    def test_search_output_exists(self):
        """
        CSV files for detailed search CV results are written as expected.
        """
        config = get_estimators_config(self.config_file)
        estimators = _init_estimators(
            config, self.categorical_variables_indices)
        data = pd.read_csv(os.path.join(self.resources_path, self.data_file))
        _search_models(data=data, estimators=estimators, config=config,
                       categorical_variables=self.categorical_variables,
                       target_variables=self.target_variables, 
                       output_path=self.output_path)
        self.assertTrue(os.path.exists(os.path.join(
            self.output_path, 
            self.results_ridge)))
        self.assertTrue(os.path.exists(os.path.join(
            self.output_path, 
            self.results_lr)))
        self.assertTrue(os.path.exists(os.path.join(
            self.output_path, 
            self.results_xgb)))

    def test_rankings_output_exists(self):
        """
        Rankings CSV file with model performance per target variable exists. 
        """
        config = get_estimators_config(self.config_file)
        estimators = _init_estimators(
            config, self.categorical_variables_indices)
        data = pd.read_csv(os.path.join(self.resources_path, self.data_file))
        stats_df, _ = _search_models(
            data=data, estimators=estimators, 
            config=config,
            categorical_variables=self.categorical_variables,
            target_variables=self.target_variables, 
            output_path=self.output_path)
        _rank_estimators(
            summary_stats=stats_df, 
            output_path=self.output_path,
            output_file=constants.AM_RANKINGS_FILE)
        self.assertTrue(os.path.exists(os.path.join(
            self.output_path,
            self.rankings)))
        
    def test_estimators_and_stuff_exists(self):
        """
        Estimators and test set results have been persisted
        """
        config = get_estimators_config(self.config_file)
        estimators = _init_estimators(
            config, cat_indices=self.categorical_variables_indices)
        data = pd.read_csv(os.path.join(self.resources_path, self.data_file))
        stats_df, estimators_per_target_variable = _search_models(
            data=data, estimators=estimators, 
            config=config,
            categorical_variables=self.categorical_variables,
            target_variables=self.target_variables, 
            output_path=self.output_path)
        rankings = _rank_estimators(
            summary_stats=stats_df, 
            output_path=self.output_path,
            output_file=constants.AM_RANKINGS_FILE)
        _select_best_estimators(
            target_variables=self.target_variables,
            rankings=rankings,
            estimators_per_target_variable=estimators_per_target_variable,
            output_path=self.output_path,
            num_jobs=config.num_jobs,
            leave_one_out_cv=None)

        # best linear and non-linear models for gpu target variable
        # print(f"******* self.output_path: {self.output_path}")
        self.assertTrue(os.path.exists(self.output_path))

        self.assertTrue(os.path.exists(os.path.join(
            self.output_path,
            self.estimator_linear)))
        self.assertTrue(os.path.exists(os.path.join(
            self.output_path,
            self.estimator_nonlinear)))

        self.assertTrue(os.path.exists(os.path.join(
            self.output_path,
            self.predictions_ridge)))
        self.assertTrue(os.path.exists(os.path.join(
            self.output_path,
            self.predictions_xgb)))
        
        self.assertTrue(os.path.exists(os.path.join(
            self.output_path,
            self.best_estimators_summary)))
        
    def test_load_and_predict(self):
        """
        Test that we can load a persisted estimator and use it to predict.
        """
        config = get_estimators_config(self.config_file)
        estimators = _init_estimators(
            config, self.categorical_variables_indices)
        data = pd.read_csv(os.path.join(self.resources_path, self.data_file))
        stats_df, estimators_per_target_variable = _search_models(
            data=data, estimators=estimators, 
            config=config,
            categorical_variables=self.categorical_variables,
            target_variables=self.target_variables, 
            output_path=self.output_path)
        rankings = _rank_estimators(
            summary_stats=stats_df, 
            output_path=self.output_path,
            output_file=constants.AM_RANKINGS_FILE)
        _select_best_estimators(
            target_variables=self.target_variables,
            rankings=rankings,
            estimators_per_target_variable=estimators_per_target_variable,
            output_path=self.output_path,
            num_jobs=config.num_jobs,
            leave_one_out_cv=None)

        test_estimator_path = os.path.join(
            self.output_path, 
            self.estimator_linear)
        loaded_estimator = joblib.load(test_estimator_path)
        X = data.drop(self.target_variables, axis=1)
        y_pred = loaded_estimator.predict(X)
        self.assertTrue(len(y_pred) > 0)
    
    def test_extrapolation_data_exists_if_feature_col(self):
        """
        Extrapolation test set created if feature column has been set.
        """
        config = get_estimators_config(self.config_file)
        estimators = _init_estimators(
                config, cat_indices=self.categorical_variables_indices)
        data = pd.read_csv(os.path.join(self.resources_path, self.data_file))
        stats_df, estimators_per_target_variable = _search_models(
            data=data, estimators=estimators, 
            config=config,
            categorical_variables=self.categorical_variables,
            target_variables=self.target_variables, 
            output_path=self.output_path,
            feature_col=self.feature_col,
            high_threshold=self.high_threshold)
        
        self.assertTrue(os.path.exists(os.path.join(
            self.output_path,
            self.extrapolation_file)))

        df = pd.read_csv(os.path.join(
            self.output_path,
            self.extrapolation_file
        ))
        self.assertGreater(len(df), 0, 
                           "Extrapolation DF should contain at least 1 row.")
        self.assertTrue(all(df[self.feature_col] >= self.high_threshold),
                            (f"All values in {self.feature_col} should be at" 
                             f" least {self.high_threshold}"))