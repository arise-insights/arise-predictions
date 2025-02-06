import unittest
import os
import sys
import logging

import pandas as pd

from arise_predictions.utils import utils, constants
from tests.utils.logger_redirector import LoggerRedirector

from arise_predictions.auto_model.build_models import (
    get_estimators_config,
    _init_estimators,
    _search_models,
    _rank_estimators,
    _select_best_estimators,
    _search_meta_model,
    _merge_rankings,
    _persist_and_test_meta_estimator
)

from arise_predictions.cmd.cmd import parse_args

"""
Tests for building of meta-learner (for improved extrapolation performance).

Note that the tearDown method deletes any temporary test files that may be
useful for debugging. Have left shutil.rmtree commented out to avoid this.
"""

logger = logging.getLogger('test-logger')

logging.basicConfig(
    format="%(asctime)s %(module)s %(levelname)s: %(message)s",
    level=logging.INFO,
    stream=sys.stdout)


class TestBuildMetaLearner(unittest.TestCase):

    def setUp(self):
        LoggerRedirector.redirect_loggers(fake_stdout=sys.stdout, 
                                          fake_stderr=sys.stderr) 

        # Simulate command-line arguments
        parse_args(["--num-jobs", "1"])

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

        self.feature_col = "# of Accelerators"
        self.high_threshold = 6
        self.extrapolation_file = "extrapolation-test-data.csv"

        self.best_estimator_name_linear_gpu = "RidgeRegression"
        self.best_estimator_name_nonlinear_gpu = "XGBoost-Regressor"
        self.best_estimator_name_linear_time = "RidgeRegression"
        self.best_estimator_name_nonlinear_time = "XGBoost-Regressor"

        self.estimator_name_meta = constants.AM_META_LEARNER_PREFIX + "ElasticNet-Regression"

        self.results_meta_learner_en = ("cv_results-meta-learner-"
                                             "ElasticNet-Regression-"
                                             "tokens_per_second.csv")
        
        self.meta_rankings_file = constants.AM_META_LEARNER_PREFIX + constants.AM_RANKINGS_FILE
        self.all_rankings_file = constants.AM_RANKINGS_FILE

        self.estimator_file_name_meta_gpu = ("estimator-linear-meta-learner-"
                                             "ElasticNet-Regression-"
                                             "gpu_memory_utilization_max.pkl")
        self.estimator_file_name_meta_time = ("estimator-linear-meta-learner-"
                                              "ElasticNet-Regression-"
                                              "train_runtime.pkl")
        self.estimator_meta_test_file = ("best-meta-estimators-testset-"
                                         "performance-summary.csv")
        self.estimator_meta_extrapolation_test_file = ("best-meta-estimators-"
                                                       "extrapolation-testset-"
                                                       "performance-summary.csv")

    def tearDown(self):
        LoggerRedirector.reset_loggers(
            fake_stdout=sys.stdout, fake_stderr=sys.stderr)
        
        # if os.path.exists(self.output_path):
        #     shutil.rmtree(self.output_path)
    
    def test_meta_learner(self):
        """
        Going against best practices and having a single method exercising all
        functionality I want to test (via separate assertion methods) to avoid
        having to run auto-model-build repeatedly (time-consuming).
        """
        config = get_estimators_config(self.config_file)
        estimators = _init_estimators(
                config, cat_indices=self.categorical_variables_indices)
        self.assertEqual(len(estimators), 4, "4 estimators")
        data = pd.read_csv(os.path.join(self.resources_path, self.data_file))

        stats_df, estimators_per_target_variable = _search_models(
            data=data, estimators=estimators, 
            config=config,
            categorical_variables=self.categorical_variables,
            target_variables=self.target_variables, 
            output_path=self.output_path,
            feature_col=self.feature_col,
            high_threshold=self.high_threshold)
        
        self._extrapolation_test_data_exists_if_feature_col()

        rankings = _rank_estimators(
            summary_stats=stats_df, 
            output_path=self.output_path,
            output_file=constants.AM_RANKINGS_FILE)

        best_estimators_for_target_variables = _select_best_estimators(
            target_variables=self.target_variables,
            rankings=rankings,
            estimators_per_target_variable=estimators_per_target_variable,
            output_path=self.output_path,
            num_jobs=config.num_jobs,
            leave_one_out_cv=None,
            categorical_variables=self.categorical_variables)
        
        self._best_estimators_as_expected(
            best_estimators_for_target_vars=best_estimators_for_target_variables)
        
        stats_df, best_meta_learner_per_target_variable = _search_meta_model(
            best_estimators_per_target_var=best_estimators_for_target_variables,
            config=config,
            cat_indices=self.categorical_variables_indices,
            target_variables=self.target_variables,
            output_path=self.output_path
        )

        self._meta_learner_results_as_expected(
            stats_df=stats_df, 
            best_meta_learner_per_target_variable=best_meta_learner_per_target_variable)
        
        _ = _rank_estimators(
            summary_stats=stats_df,
            output_path=self.output_path,
            output_file=self.meta_rankings_file)
        
        _merge_rankings(output_path=self.output_path) 
        
        self._meta_learner_rankings_exists()

        _persist_and_test_meta_estimator(
            meta_learner_per_target_variable=best_meta_learner_per_target_variable,
            output_path=self.output_path,
            num_jobs=config.num_jobs,
            leave_one_out_cv=None,
            categorical_variables=self.categorical_variables
        )

    def _extrapolation_test_data_exists_if_feature_col(self):
        """
        Extrapolation test data exists if feature column has been set.
        """
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
    
    def _best_estimators_as_expected(self, best_estimators_for_target_vars):
        """
        List of best estimators is as expected.
        """
        self.assertEqual(len(best_estimators_for_target_vars), 1,
                         "One target variable")
        best_estimators_for_thpt = best_estimators_for_target_vars[
            self.target_variables[0]]
        self.assertEqual(len(best_estimators_for_thpt), 2,
                         "2 best estimators per target variable")
        estimator_names_thpt = [estimator["estimator_name"] for estimator in best_estimators_for_thpt]
        self.assertIn(self.best_estimator_name_linear_gpu, estimator_names_thpt)
        self.assertIn(self.best_estimator_name_nonlinear_gpu, estimator_names_thpt)

    def _meta_learner_results_as_expected(self, stats_df, best_meta_learner_per_target_variable):
        """
        Meta learner CV results exist and have one meta-learner per target variable.
        """
        self.assertTrue(os.path.exists(os.path.join(
            self.output_path,
            self.results_meta_learner_en
        )))
        self.assertEquals(
            len(best_meta_learner_per_target_variable[self.target_variables[0]]),
            1, 
            "1 meta-learner per target variable"
        )
        utils.write_df_to_csv(stats_df, self.output_path, "A_STATS_DF_INSPECT.csv")

    def _meta_learner_rankings_exists(self):
        """
        The meta-all-star-rankings exists and is data is in all-star-rankings.
        """
        self.assertTrue(os.path.exists(os.path.join(
            self.output_path,
            self.meta_rankings_file 
        )))
        self.assertTrue(os.path.exists(os.path.join(
            self.output_path,
            self.all_rankings_file 
        ))) 
        rankings_df = pd.read_csv(os.path.join(
            self.output_path, 
            self.all_rankings_file
            )
        )
        self.assertIn(
            self.estimator_name_meta, 
            rankings_df[constants.AM_COL_ESTIMATOR].tolist())

    def _meta_learner_persisted_and_tested(self):
        """
        Meta-learner has been persisted along with test results.
        """
        self.assertTrue(os.path.exists(os.path.join(
            self.output_path,
            self.estimator_file_name_meta_gpu
        )))
        self.assertTrue(os.path.exists(os.path.join(
            self.output_path,
            self.estimator_file_name_meta_time
        )))
        self.assertTrue(os.path.exists(os.path.join(
            self.output_path,
            self.estimator_meta_test_file
        )))
        self.assertTrue(os.path.exists(os.path.join(
            self.output_path,
            self.estimator_meta_extrapolation_test_file
        )))
