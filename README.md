# Comparison of cost sensitive learning algorithms and sampling methods on imbalanced datasets

There are many applications which require dealing with imbalanced datasets, such as fraud detection, fault prediction in industrial machinery and rare diseases diagnostic. Machine learning algorithms can tend to favor the majority class error minimization, which is frequently of less interest in practice. Approaches to overcome this difficulty include data resampling and the use of cost sensitive algorithms. These algorithms tend to be less studied due to the requirement to define a cost matrix, which can be determined by a specialist or via techniques to estimate its values. Nonetheless, cost sensitive algorithms can have less impact on the dataset than resampling methods. This work aims to compare the performance of these approaches on datasets with varying levels of imbalance. 

### This repository is structured as follows:

* **synthetic_datasets**:
* * scripts: contains the scripts to create the synthetic datasets
* * output: contains the synthetic datasets
* **experiments**:
* * `run_pmlb.py`: script to run experiments on Penn Machine Learning Benchmark
* * `run_synthetic_datasets.py`: script to run experiments on synthetic datasets
* * `config.py`: experiments hyperparameters
* * `experiment.py`: data processing pipeline
* * `BayesMinimumRiskClassifierSklearnCompat.py`: Wrapper to make costcla's BayesMinimumRisk classifier compatible with Scikit-learn cross-validation
* * `ThresholdingSklearnCompat.py`: Wrapper to make costcla's Thresholding classifier compatible with Scikit-learn cross-validation
* * `CostSensitiveDecisionTreeSklearnCompat.py`: Wrapper to make costcla's Cost Sensitive Decisiion Tree classifier compatible with Scikit-learn cross-validation
* * `cost_matrix_util.py` Cost matrix auxiliary functions

## Synthetic datasets

Each dataset contains *N=10,000* examples with two predictors (*v=2*) and one binary target variable. The majority variable was obtained from an uniform distribution e covers the whole available space. 
