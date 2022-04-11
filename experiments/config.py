
import numpy as np

#Scalers
from sklearn.preprocessing import MinMaxScaler, RobustScaler

#Algorithms for models
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

#Resampling algorithms
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

#Cost Sensitive Learning algorithms
from ThresholdingSklearnCompat import Thresholding
from BayesMinimumRiskClassifierSklearnCompat import BayesMinimumRisk
from CostSensitiveDecisionTreeSklearnCompat import CostSensitiveDT

imbalance_setup_dict = {
    'None': {
        'class': None,
        'algorithm': None,
        'hyperparameters': {None: [None]} }, 
    'SMOTE': {
        'class': 'resampling',
        'algorithm': SMOTE,
        'hyperparameters': {
            'sampling_strategy': ['minority'],
            'k_neighbors': [3, 5, 7],
            'random_state': [324089] } },
    'SMOTE Tomek': {
        'class': 'resampling',
        'algorithm': SMOTETomek,
        'hyperparameters': {
            'sampling_strategy': ['minority'],
            'smote': [SMOTE(k_neighbors=x) for x in [3, 5, 7]],
            'random_state': [324089] } },
    'Tomek Links': {
        'class': 'resampling',
        'algorithm': TomekLinks,
        'hyperparameters': {
            'sampling_strategy': ['majority'] } },
    'Thresholding': {
        'class': 'cost-sensitive learning',
        'algorithm': Thresholding,
        'hyperparameters': {
            'C': [(1,1), (1,2), (1,5), (1,10), 'balanced'] } },
    'Bayes Minimum Risk': {
        'class': 'cost-sensitive learning',
        'algorithm': BayesMinimumRisk,
        'hyperparameters': {
            'C': [(1,1), (1,2), (1,5), (1,10), 'balanced'] } }
}

classifiers_setup_dict = {
    'Cost Sensitive Decision Tree': {
        'classifier': CostSensitiveDT,
        'scaler': RobustScaler,
        'hyperparameters': {
            'cost': [(1,1), (1,2), (1,5), (1,10), 'balanced'],
            'max_features': [0.1, 0.25, 0.5, 0.75, 'sqrt', 'log2', None],
            'criterion': ['direct_cost', 'gini_cost', 'entropy_cost'],
            'min_gain': np.arange(0., 0.005, 0.00025)} }, 
    'Decision Tree': {
        'classifier': DecisionTreeClassifier,
        'scaler': RobustScaler,
        'hyperparameters': {
            'min_impurity_decrease': np.arange(0., 0.005, 0.00025),
            'max_features': [0.1, 0.25, 0.5, 0.75, 'sqrt', 'log2', None],
            'criterion': ['gini', 'entropy'],
            'random_state': [324089] } }, 
    'Random Forest': {
        'classifier': RandomForestClassifier,
        'scaler': RobustScaler,
        'hyperparameters': {
            'n_estimators': [10, 50, 100, 500],
            'min_impurity_decrease': np.arange(0., 0.005, 0.00025),
            'max_features': [0.1, 0.25, 0.5, 0.75, 'sqrt', 'log2', None],
            'criterion': ['gini', 'entropy'],
            'random_state': [324089] } }, 
    'KNN': {
        'classifier': KNeighborsClassifier,
        'scaler': RobustScaler,
        'hyperparameters': {
            'n_neighbors': list(range(1, 26)) + [50, 100],
            'weights': ['uniform', 'distance'] } },
    'AdaBoost': {
        'classifier': AdaBoostClassifier,
        'scaler': RobustScaler,
        'hyperparameters': {
            'n_estimators': [10, 50, 100, 500],
            'learning_rate': [0.01, 0.1, 0.5, 1.0, 10.0, 50.0, 100.0],
            'random_state': [324089] } }
}

if __name__=='__init__':
    EXPECTED_NUMBER_CLASSIFIERS = 5
    #Test classifier has the expected number of classifiers
    assert len(classifiers_setup_dict) == EXPECTED_NUMBER_CLASSIFIERS
    #Test that every classifier have the expected objects used by pipeline
    for classifier in classifiers_setup_dict:
        assert {'classifier', 'scaler', 'hyperparameters'} == set(classifier.keys())
    