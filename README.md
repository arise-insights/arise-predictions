# arise-predictions: AI Right Sizing Engine


1. [Overview](#overview)
2. [Installing and running the CLI](#installing-and-running-the-cli)
3. [ARISE in action on sample data](#arise-in-action-on-sample-data)
4. [Running ARISE from the UI - Work in Progress](#running-arise-from-the-ui)
5. [More on data requirements](#historical-data)
6. [Known tool issues](#known-tool-issues)


## Overview

AI Right Sizing Engine (ARISE) is a tool for predicting required resources and execution time of an AI workload, based 
on historical executions or performance benchmarks of similar workloads (a workload dataset). ARISE is intended to 
support configuration decision-making for platform engineers or data scientists operating with AI stacks. 

ARISE parses and preprocesses the given workloads dataset into a standard format, provides descriptive statistics, 
trains predictive models, and performs predictions based on the models. See [Instructions for running the CLI](#installing-and-running-the-cli) for 
details on the commands to invoke the above operations. To use these commands, in addition to the workload dataset, you 
need to provide in your input path a `job_spec.yaml` file indicating the metadata inputs and outputs of your data.
See [this example](examples/MLCommons/job_spec.yaml) of a job spec.

## Installing and running the CLI

### Installing from a repo snapshot

- Clone the repo or download codebase zip

- Install the CLI

To install the CLI in a virtual environment (this would be the preferred installation mode to keep the 
installation isolated and avoid version conflicts), run the commands:

```buildoutcfg
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Windows users should run:

```buildoutcfg
python3 -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

### Execute tests

From the project root directory:

```bash
python -m unittest -v
```

To see the log messages for failing tests, use the buffer command (and use
`tests.utils.logger_redirector` in your test case). See `tests.test_analyze.py`
as an example.

```bash
python -m unittest -v --buffer
```

Running the tests on a single test case:

```bash
python -m unittest -v --buffer tests/test_build_models.py
```

### Running the CLI on sample data

There are four supported commands:

1. `analyze-jobs` provides descriptive statistics on the metadata inputs (workload
measurements) and generates a number of spreadsheets and plots in a
subdirectory called `job-analysis`. The data should be provided in a folder
called `data` in the given `input-path`. To use this command, you need to provide in your input path 
a `job_spec.yaml` file indicating the metadata inputs and outputs of your data. See [this example](examples/MLCommons/job_spec.yaml) of a job spec.

```bash
python -m arise_predictions.main analyze-jobs --input-path examples/MLCommons
```

It is also possible to specify the metadata input explicitly:

```bash
python -m arise_predictions.main.py analyze-jobs --input-path examples/MLCommons --reread-history --input-file inference_data_tokens.csv --custom-job-name inference-thpt
```

In the above example, we also specify a custom job name. In this example data
set there is no column capturing the job id. If there were, we could provide it in the `--job-id-column` argument. With 
`--custom-job-name`, we instruct the code to insert such a column with the given job name as values.  
This tends to improve the output of the descriptive job analysis  (e.g., labels in plots).

2. `auto-build-models` performs a hyperparameter search over the models and
   parameter space specified in a configuration file (cf.,
   `config/default-auto-model-search-config.yaml`) and finds the best model and
   its hyperparameter settings for each target variable in the data. It attempts
   to build one best model per target variable in the metadata outputs based on
   the metadata inputs. To use this command, you need to provide in your input path 
   a `job_spec.yaml` file indicating the metadata inputs and outputs of your data. 
   See [this example](examples/MLCommons/job_spec.yaml) of a job spec.

Example:

```bash
python -m arise_predictions.main auto-build-models --input-path examples/MLCommons --reread-history
```

The output models, their relative ranking, and the cross validation results are all stored in a folder named 
`ARISE-auto-models` which is created in the given input path. If you run with the flag `--single-output-file`, the
models and results will be archived into a single output file `ARISE-auto-models.zip` in the given input path.

If you do not specify an option for `--config-file`, it uses the default one in
`config/default-auto-model-search-config.yaml`. There is a config file that
defines a much smaller parameter search space and hence completes in a shorter
time. You can make use of it like this:

```bash
python -m arise_predictions.main auto-build-models --input-path examples/MLCommons --reread-history --config-file config/small-auto-model-search-config.yaml
```

If you are running on your local machine, it is advised to limit the number of processors used. However, this will result
in a much longer run. To build models using 2 processors only, use this command: 

```bash
python -m arise_predictions.main --num-jobs 2 auto-build-models --input-path examples/MLCommons --reread-history --config-file config/small-auto-model-search-config.yaml
```

By default, `auto-build-models` performs 10-fold cross validation. If you want to perform instead `leave-one-group-out (logo)`
cross validation, add the cflag `--leave-one-out-cv`, which takes as an argument a list of one or more feature names to 
group py, separated by commas.

For example, the following command will build models using logo cross validation on values of LLM name.
That is, in each iteration it will use a specific LLM as the test set, and all other data as the training set.

```bash
python -m arise_predictions.main auto-build-models --input-path examples/MLCommons --leave-one-out-cv "Model MLC"
```

In addition to the above, we can also let `auto-build-models` search for models that are tuned for
extrapolation. That is, you can let ARISE build a model that performs
*relatively* well when asked to predict on inputs that are outside the range of
values seen for this feature during training. This is an experimental feature
whose performance we expect to improve with time. Currently, only a single
extrapolated feature is supported and the training data needs a number of
different data points or levels for this feature to have an opportunity to learn
to extrapolate.
You specify the name of the input feature on which to extrapolate
(`--feature-column`) as well as a low and or high threshold (`--low-threshold`,
`--high-treshold`) which define the extrapolation region to train on. The
thresholds should be chosen from within the range of values that exist in the
training data, so that ARISE can define regions used for training and for
testing the extrapolation performance of the resulting models. For example: 

```bash
python -m arise_predictions.main auto-build-models --input-path examples/MLCommons --reread-history --feature-column "# of Accelerators" --high-threshold 8
```

3. `predict` generates estimated values for metadata outputs given metadata input values. It should 
be run after `auto-build-models` command and uses its output. The `--model-path` flag is where the models created by `auto-build-models` are located.
Predict requires to specify a model name and input space configuration.
It generates the space of input features according to the configuration and uses models previously built with `auto-build-models` to run predictions on this
input space for the target variables indicated in the same configuration file.

An example configuration file: [example-demo-mlcommons-config.yaml](config/example-demo-mlcommons-config.yaml).

In addition to feature values, the config file requires specifying target variables for prediction. For each target 
variable, the boolean parameter indicating whether `greater_is_better` should be specified. `estimator_file` is an 
optional parameter providing the name of the model file to use for predictions of this target variable. If it is not 
provided, ARISE automatically uses the top-ranked model file according to the `auto-build-models` results which should 
be located in the provided model path, next to the persisted model files.

```bash
python -m arise_predictions.main predict --input-path examples/MLCommons --config-file
config/example-demo-mlcommons-config.yaml --model-path examples/MLCommons/ARISE-auto-models
```

The input space defined by the configuration file and ARISE predictions for each input combination in this space are  
stored in a folder named `ARISE-predictions` which is created in the given input path.

4. `demo-predict` is a version of predict that facilitates demos by ranking
    predictions and comparing predictions with ground truth where available. 
    It also enables obtaining prediction values from the given data instead 
    of specifying them explicitly.


    The `--input-path` should point to historic or benchmark input data so `demo-predict` could 
    compare predictions with available ground truth (as far as is possible). The script needs to have the path to
    the directory containing the serialized models built by `auto-build-models`.
    Other parameters are taken from the configuration file.

```bash
python -m arise_predictions.main demo-predict --input-path examples/MLCommons --config-file
config/example-demo-mlcommons-demo-predict-config.yaml --model-path examples/MLCommons/ARISE-auto-models
```

In addition to the outputs described for the `predict` command, `demo-predict` will also create a file named 
`predictions-with-ground-truth.csv`, containing the predicted versus ground truth values and the resulting MAPE error, 
for any input combination in the defined input space that appears also in the given ground truth data. 

Note that [a different configuration file](config/example-demo-mlcommons-demo-predict-config.yaml) was used for 
`demo-predict` than the one used for `predict`. It includes a `data_values` list. Rather than explicitly listing
the values to be predicted as in the `predict` configuration file, the values are taken from the given data. 
The `values` key can be either `all` for taking all values from the data, or `min_max` for taking the entire range 
from minimal to maximal value appearing in the data (the latter is applicable to numeric inputs only). You can also 
specify a list of values to exclude from prediction. In our example, the values for `Accelerator` are instructed to be 
taken from the data, the values for `# of Accelerators` are instructed to spread from the minimal to maximal value 
appearing in the data (this is of course possible for numeric inputs only), and the case `# of Accelerators = 0` is 
excluded from the prediction space. If the same input appears also in the `variable_values` list, as in the case of 
`# of Accelerators`, the values explicitly specified (`9` in our example) are added to the values derived from the data.

5. `data-predict` is a version of predict that receives the prediction space directly as a dataframe (read from a 
   csv file given in `--prediction-data-file`) instead of defining it via a configuration file. The configuration
   file is still provided, just for specifying properties of the target variables (i.e., the estimators section). If an 
   original data file is provided (in `--original-data-file`), ground truth is calculated by comparing predicted 
   outputs to the outputs that appear in it. If the flag `--delta-only` is provided and the original data is provided 
   as well, predictions are performed only for input combinations that appear in the prediction file but not in the  
   original data. This is useful, for example, if the original data provided is the training data, and we want to 
   predict only for input combinations unseen by the model during training.


The default log level is `DEBUG`. You can change by specifying a different log
level as in the following example:

```bash
python -m arise_predictions.main --loglevel info analyze-jobs
```

## Running ARISE from the UI

To run ARISE from the UI, see documentation [here](ui/README.md). Note that the UI is still work in progress and missing
many features that are available from the CLI.

## ARISE in Action on Sample Data

To see ARISE in action on a sample dataset, go [here](doc/example.md).

## Historical Data

The data consists of historical workload executions and/or performance benchmarks. Examples of potential properties of 
workloads that can be considered:

1. Input data size and data complexity-related properties
2. Hyper-parameters
3. Workload task
4. LLM
5. GPU configuration
6. Total execution time 
7. Throughput and latency
8. Consumed resources: number of workers, CPU, GPU, and memory per worker
9. Job status (success, fail/abort, etc.)

Example datasets can be found [here](examples/MLCommons/data) and [here](examples/sentiment_analysis/data).

The data is divided into `job-metadata-inputs`: the properties of the workload that are known before it starts running 
(e.g., items 1-5 above), and `job-metadata-outputs`: properties of the workload execution and output that are known only 
once the workload completes (e.g., items 6-9 above). The inputs and outputs specification is provided in the 
`job_spec.yaml` file. See [this example](examples/MLCommons/job_spec.yaml) of a job spec.

In your job spec, you can use the `job-entry-filter` key to filter out entries from the original data according to 
specific input values. In [this example](examples/MLCommons/job_spec_with_value_filter.yaml), we filter out all entries 
where the Processor is`2xAMD EPYC 9374F`, but we keep Processor as a data input. The semantics between the different 
entries specified in `job-entry-filter` is OR. That is, an entry matching any of the values specified will be 
filtered out.

If the format of your data requires special parsing to transform into a dataframe (i.e., beyond a simple csv file), you 
can implement your own parser in [this class](arise_predictions/preprocessing/custom_job_parser.py). For example, the sentiment 
analysis example ([here](examples/sentiment_analysis/data)) uses `SAJsonJobParser` as its parser, since its original 
data consists of a json file per workload execution. The name of your parser should be provided in the 
`job-parser-class-name` optional field in `job_spec.yaml`, see [here](examples/sentiment_analysis/job_spec.yaml).

## Known Tool Issues

1. Currently, the tool uses exhaustive grid search for hyperparameter optimization (HPO). This may result in long run 
time for large datasets. We plan to move to a sample-based HPO that will scale the model search phase.
2. Extrapolation is still work in progress, hence currently we expect large errors when predicting outputs for input 
values which are far beyond the range provided in the training dataset.