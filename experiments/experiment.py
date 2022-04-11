import pandas as pd
import numpy as np
from os import path
import itertools
from datetime import datetime 
import tqdm
from config import imbalance_setup_dict, classifiers_setup_dict

import warnings
warnings.filterwarnings("ignore")

from multiprocessing import Pool


#Model selection and metrics
from sklearn.model_selection import train_test_split, cross_val_predict, cross_validate, cross_val_score, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

class Experiment():

    def set_experiment_name(self, experiment_name):
        assert type(experiment_name)==str
        self.name = experiment_name

    def set_log_directory(self, log_directory):
        assert type(log_directory)==str, "log_directory is not a string"
        assert path.exists(log_directory), "log directory does not exist"
        self.log_directory = log_directory

    def config_log(self):
        self.log_metadata={'experiment_name': str, 'metadata': dict, 'train_size': float, 
                           'imbalance_approach': str, 'imbalance_approach_hyperparams': dict,
                           'learning_algorithm': str, 'learning_algorithm_hyperparams': dict,
                           'test_balanced_accuracy': np.float64, 
                           'test_precision': np.float64,
                           'test_recall': np.float64, 
                           'test_f1_macro': np.float64, 
                           'test_roc_auc': np.float64, 
                           'test_confusion_matrix': np.ndarray,
                           'fit_time': float}

    def check_log(self, log):
        assert type(log)==dict, "log must be a dict"
        assert set(log.keys()) == set(self.log_metadata.keys()), "missing or exceding fields in log"
        for attr in log.keys():
            assert type(log[attr]) == self.log_metadata[attr], "{} must be {}".format(attr, self.log_metadata[attr])
        return log        

    def save_experiment_logs(self, logs, verbose=True):
        logs_df = pd.DataFrame(logs, columns=self.log_metadata.keys())
        logs_df.to_csv(self.log_directory+'/'+self.name)
        if verbose:
            print("Experiment {} finished successfully. Output has shape {}.".format(self.name, logs_df.shape))

    def set_XY(self, df, target):
        assert type(df)==pd.DataFrame, "df is not a Pandas DataFrame"
        assert type(target)==str, "target is not a string"
        assert target in df.columns, "target is not in dataframe"

        self.X = df.loc[:, df.columns != target]
        self.y = df[target]

    def set_metadata(self, metadata):
        if metadata!=None:
            assert type(metadata) == type({})
            self.metadata = metadata
        else:
            self.metadata = dict()

    def set_train_size(self, train_size):
        assert type(train_size)==float, "train_size is not a float"
        assert train_size>0.0 and train_size<1.0, "train_size is not in range (0.0, 1.0)"
        self.train_size = train_size

    def set_n_splits_cv(self, n_splits_cv):
        assert type(n_splits_cv)==int, "n_splits_cv is not int"
        assert n_splits_cv>1, "n_splits_cv is less than 2"
        self.n_splits_cv = n_splits_cv

    def set_scoring(self, scoring):
        assert type(scoring)==list, "scoring must be a list of at least 1 score metric"
        self.scoring = scoring

    def set_maximization_metric(self, maximize_metric):
        self.maximize_metric = maximize_metric

    def split_dataset(self, random_state=None):
        """
        Split dataset in train and test
        -----------------------------------------------------------------------------------------
        Input:
        random_state: Optional seed for random state
        -----------------------------------------------------------------------------------------
        Output: None (result is saved inside object)
        """
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, 
                                                            train_size=self.train_size, 
                                                            stratify=self.y, 
                                                            random_state=random_state, 
                                                            shuffle=True)
        self.X_train = X_train.values
        self.y_train = y_train.values
        self.X_test = X_test.values
        self.y_test = y_test.values

    def scale_dataset(self, scalers):
        """
        Since most experiments share the algorithm chosen to scale the data, this method perform
        this operation once and store the scaled data for every scaler used.
        -----------------------------------------------------------------------------------------
        Input:
        scalers: All scalers used
        -----------------------------------------------------------------------------------------
        Output: None (result is saved inside object)
        """
        self.X_train_scaled = {}
        self.X_test_scaled = {}
        self.X_train_scaled_df = {}
        self.X_test_scaled_df = {}
        for scaler in scalers:
            fitted_scaler = scaler().fit(self.X_train)
            self.X_train_scaled[scaler] = fitted_scaler.transform(self.X_train)
            self.X_test_scaled[scaler] = fitted_scaler.transform(self.X_test)

            #self.X_train_scaled_df[scaler] = pd.DataFrame(self.X_train_scaled[scaler],
            #                                            columns=self.X.columns)
            #self.X_test_scaled_df[scaler] = pd.DataFrame(self.X_test_scaled[scaler],
            #                                            columns=self.X.columns)

            #assert self.X_train_scaled_df[scaler].merge(self.y_train, left_index=True, right_index=True).isnull().sum().sum()==0
            #assert self.X_train_scaled_df[scaler].merge(self.y_test, left_index=True, right_index=True).isnull().sum().sum()==0
  
    def kfold_cv(self, imbalance_approach_class, imbalance_approach, 
                 imbalance_approach_hyperparameters, learning_algorithm, scaler, 
                 learning_algorithm_hyperparameters, X, y):

        """
        Fit classifier to experiment data and return cross validation score
        -----------------------------------------------------------------------------------------
        Input:
        imbalance_approach_class: One of ('resampling', 'cost-sensitive learning', None)
        imbalance_approach: Meta-algorithm used to make the classifier cost-sensitive
        imbalance_approach_hyperparameters: Hyperparameters of given meta-algorithm (dict)
        learning_algorithm: Classifier
        scaler: Scaler used to rescale the data
        learning_algorithm_hyperparameters: Hyperparameters of given classifier (dict)
        X: Predictors
        y: Target variable
        -----------------------------------------------------------------------------------------
        Output: 
        Input (except X,y) appended with cross validation score
        """

        #Not that base classifier was already set up upon passing to Thresholding
        #so fit_params will be ommited for this special case
        # if type(imbalance_approach_class)!=type(None):
        #     if imbalance_approach.__name__=='Thresholding' or imbalance_approach.__name__=='BayesMinimumRisk':
        #         fit_params = {}
        #     else:
        #         fit_params = learning_algorithm_hyperparameters
        # else:
        #     fit_params = learning_algorithm_hyperparameters
        fit_params = learning_algorithm_hyperparameters
        try:
            cv = StratifiedKFold(n_splits=self.n_splits_cv, shuffle=True, random_state=90483257)
            pre_mean=cross_val_score(learning_algorithm(**fit_params), X=X, y=y, scoring=self.maximize_metric, cv=cv)
            cv_score = np.nan_to_num(np.mean(pre_mean))

            return [imbalance_approach_class, imbalance_approach, imbalance_approach_hyperparameters, 
                    learning_algorithm, scaler, learning_algorithm_hyperparameters, cv_score]
        except:
            return [imbalance_approach_class, imbalance_approach, imbalance_approach_hyperparameters, 
                    learning_algorithm, scaler, learning_algorithm_hyperparameters, 0.0]

    def select_best_hyperparameters(self, cross_validations):
        """
        Select best hyperparameter given cross validation performance
        -----------------------------------------------------------------------------------------
        Input:
        cross_validations: Array of learning_algorithm, hyperparameters, cv_score
        -----------------------------------------------------------------------------------------
        Output: 
        best_params: Array with imbalance_approach_class, imbalance_approach, 
                     imbalance_approach_hyperparameters, learning_algorithm, scaler, 
                     learning_algorithm_hyperparameters
        """
        best_i = np.argmax([cv[6] for cv in cross_validations])
        best_params = cross_validations[best_i][0:6]
        return best_params

    def resample(self, imbalance_approach, imbalance_approach_hyperparameters, X, y):
        """
        Resample data according to imbalanche_approach
        -----------------------------------------------------------------------------------------
        Input:
        imbalance_approach: Algorithm used to resample
        imbalance_approach_hyperparameters: Hyperparameters of given algorithm (dict)
        X: Predictors
        y: Target variable
        -----------------------------------------------------------------------------------------
        Output: 
        X_res, y_res: Predictor and target variable of resampled data
        """
        if len(imbalance_approach_hyperparameters)>0:
            res = imbalance_approach(**imbalance_approach_hyperparameters)
        else:
            res = imbalance_approach()
        X_res, y_res = res.fit_resample(X, y)
        return X_res, y_res

    def cost_sensitive_learning(self, learning_algorithm, learning_algorithm_hyperparameters,
                                imbalance_approach, imbalance_approach_hyperparameters):
        """
        Transform a base learner classifier into cost-sensitive  
        -----------------------------------------------------------------------------------------
        Input:
        learning_algorithm: Classifier
        learning_algorithm_hyperparameters: Hyperparameters of given classifier (dict)
        imbalance_approach: Meta-algorithm used to make the classifier cost-sensitive
        imbalance_approach_hyperparameters: Hyperparameters of given meta-algorithm (dict)
        -----------------------------------------------------------------------------------------
        Output: 
        cost_sensitive_clf: Classifier transformed into cost-sensitive
        """
        #Set base_classifier
        if len(learning_algorithm_hyperparameters)>0:
            base_clf = learning_algorithm(**learning_algorithm_hyperparameters)
        else:
            base_clf = learning_algorithm()

        if imbalance_approach.__name__ == 'MetaCost':
            #Metacost interface
            m = imbalance_approach_hyperparameters['m']
            cost_matrix = imbalance_approach_hyperparameters['C']
            p = 'predict_proba' in dir(learning_algorithm)
            cost_sensitive_clf = imbalance_approach(base_clf, cost_matrix, m, use_predict_proba=p)
            return  cost_sensitive_clf

        elif imbalance_approach.__name__=='Thresholding' or imbalance_approach.__name__=='BayesMinimumRisk':
            #Costcla interface
            cost_matrix = imbalance_approach_hyperparameters['C']
            cost_sensitive_clf = imbalance_approach(base_clf, cost_matrix)
            return cost_sensitive_clf
        
        else:
            raise ValueError('Invalid imbalanced_approach')
                     
    def fit_score_cv(self, params):
        (imbalance_approach_class, imbalance_approach, imbalance_approach_hyperparameters, 
         learning_algorithm, scaler, learning_algorithm_hyperparameters) = params
        if imbalance_approach==None:
            cv_eval = self.kfold_cv(imbalance_approach_class, imbalance_approach, 
                                    imbalance_approach_hyperparameters, learning_algorithm, scaler, 
                                    learning_algorithm_hyperparameters,
                                    self.X_train_scaled[scaler], self.y_train)

        elif imbalance_approach_class=='resampling':
            X_res, y_res = self.resample(imbalance_approach, imbalance_approach_hyperparameters,
                                        self.X_train_scaled[scaler], self.y_train)
            cv_eval = self.kfold_cv(imbalance_approach_class, imbalance_approach, 
                                    imbalance_approach_hyperparameters, learning_algorithm, scaler, 
                                    learning_algorithm_hyperparameters,
                                    X_res, y_res)

        elif imbalance_approach_class=='cost-sensitive learning':
            cost_sensitive_clf = self.cost_sensitive_learning(learning_algorithm, 
                                                              learning_algorithm_hyperparameters,
                                                              imbalance_approach, 
                                                              imbalance_approach_hyperparameters)
            cv_eval = self.kfold_cv(imbalance_approach_class, imbalance_approach, 
                                    imbalance_approach_hyperparameters, cost_sensitive_clf, scaler, 
                                    learning_algorithm_hyperparameters, self.X_train_scaled[scaler], 
                                    self.y_train)
        else:
            raise ValueError("Imbalance approach class '{}' not supported".format(imbalance_approach_class))
        
        return cv_eval

    def select_hyperparameters(self, selection_grid, n_jobs=None):
        pool = Pool(n_jobs)
        cv_evals = pool.map(self.fit_score_cv, selection_grid)
        pool.close()
        pool.join()
        best_params = self.select_best_hyperparameters(cv_evals)
        return best_params

    def predict_test(self, params):
        (imbalance_approach_class, imbalance_approach, imbalance_approach_hyperparameters, 
         learning_algorithm, scaler, learning_algorithm_hyperparameters) = params
        
        if imbalance_approach==None:
            t_start = datetime.now()
            clf_fitted = learning_algorithm(**learning_algorithm_hyperparameters)
            clf_fitted.fit(self.X_train_scaled[scaler], self.y_train)
            t_end = datetime.now()

        elif imbalance_approach_class=='resampling':
            t_start = datetime.now()
            X_res, y_res = self.resample(imbalance_approach, imbalance_approach_hyperparameters,
                                        self.X_train_scaled[scaler], self.y_train)
            clf_fitted = learning_algorithm(**learning_algorithm_hyperparameters
                                            ).fit(X_res, y_res)
            t_end = datetime.now()

        elif imbalance_approach_class=='cost-sensitive learning':
            t_start = datetime.now()
            cost_sensitive_clf = self.cost_sensitive_learning(learning_algorithm.get_classifier_contructor(), 
                                                              learning_algorithm_hyperparameters,
                                                              imbalance_approach, 
                                                              imbalance_approach_hyperparameters)
            clf_fitted = cost_sensitive_clf.fit(self.X_train_scaled[scaler], self.y_train)
            t_end = datetime.now()
        
        
        y_pred = clf_fitted.predict(self.X_test_scaled[scaler])
        test_balanced_accuracy = balanced_accuracy_score(self.y_test, y_pred)
        test_precision = precision_score(self.y_test, y_pred)
        test_recall = recall_score(self.y_test, y_pred)
        test_f1_macro = f1_score(self.y_test, y_pred, average='macro')
        test_roc_auc = roc_auc_score(self.y_test, y_pred)
        test_confusion_matrix = confusion_matrix(self.y_test, y_pred)
        fit_time = (t_end-t_start).total_seconds()
        
        return (imbalance_approach_class, imbalance_approach, imbalance_approach_hyperparameters, 
                learning_algorithm, scaler, learning_algorithm_hyperparameters, 
                test_balanced_accuracy, test_precision, test_recall, test_f1_macro,
                test_roc_auc, test_confusion_matrix, fit_time)
        
    def train_and_evaluate(self, scaler, learning_algorithm, learning_algorithm_hyperparameters, 
                            imbalance_approach, imbalance_approach_class, 
                            imbalance_approach_hyperparameters):
        """
        Train and evaluate a classifier and an imbalanced_approach usign cross-validation
        -----------------------------------------------------------------------------------------
        Input:
        scaler: Scaler used to rescale the data
        learning_algorithm: Classifier
        learning_algorithm_hyperparameters: Hyperparameters of given classifier (dict)
        imbalance_approach: Meta-algorithm used to make the classifier cost-sensitive
        imbalance_approach_hyperparameters: Hyperparameters of given meta-algorithm (dict)
        -----------------------------------------------------------------------------------------
        Output: 
        cv_report: Time taken, train and test performance evaluated using cross-validation
        """
        if imbalance_approach==None:
            cv_report = self.fit_learning_algorithm(learning_algorithm, 
                                                    learning_algorithm_hyperparameters,
                                                    self.X_train_scaled[scaler], 
                                                    self.y_train)

        elif imbalance_approach_class=='resampling':
            X_res, y_res = self.resample(imbalance_approach, imbalance_approach_hyperparameters,
                                        self.X_train_scaled[scaler], self.y_train)
            cv_report = self.fit_learning_algorithm(learning_algorithm, 
                                                    learning_algorithm_hyperparameters,
                                                    X_res, y_res)

        elif imbalance_approach_class=='cost-sensitive learning':
            cost_sensitive_clf = self.cost_sensitive_learning(learning_algorithm, 
                                                              learning_algorithm_hyperparameters,
                                                              imbalance_approach, 
                                                              imbalance_approach_hyperparameters)
            cv_report = self.fit_learning_algorithm(cost_sensitive_clf, {}, 
                                                self.X_train_scaled[scaler], 
                                                self.y_train)
        else:
            raise ValueError("Imbalance approach class '{}' not supported".format(imbalance_approach_class))

        return cv_report

    def create_params_grid(self):
        """
        Create configurations for experiment runs by applying cartesian product to 
        experiments_configuration
        -----------------------------------------------------------------------------------------
        Input:
        -----------------------------------------------------------------------------------------
        Output: 
        grid: Array with configurations for every run of the experiment
        """
        grid = []

        for imb_name in imbalance_setup_dict.keys():
            imb = imbalance_setup_dict[imb_name]
            #Cartesian product of imabalance approach hyperparameters
            cart_prod_imb_hyp = list(itertools.product(*imb['hyperparameters'].values()))
            for imb_hyp in cart_prod_imb_hyp:
                #Transform list of hyperparameters into dict
                imb_hyp_keys = list(imb['hyperparameters'].keys())
                imb_hyp = {imb_hyp_keys[i]: imb_hyp[i] for i in range(len(imb_hyp_keys))}
                for clf_name in list(classifiers_setup_dict.keys()):
                    clf = classifiers_setup_dict[clf_name]
                    #Cartesian product of classifier hyperparameters
                    cart_prod_clf_hyp = list(itertools.product(*clf['hyperparameters'].values()))
                    #Transform list of hyperparameters into dict
                    clf_hyp_keys = list(clf['hyperparameters'].keys())
                
                    clf_hyp = [{clf_hyp_keys[i]: hyp[i] for i in range(len(clf_hyp_keys)) } 
                                                        for hyp in cart_prod_clf_hyp]                    
                    if clf_name=='Logistic Regression':
                        clf_hyp = [x for x in clf_hyp if x['penalty']=='l2' or x['dual']==False]
                    
                                    
                    mini_grid = list(itertools.product([imb['class']], [imb['algorithm']], [imb_hyp],
                                                       [clf['classifier']], [clf['scaler']], clf_hyp))
                    grid.append(mini_grid)
        return grid         
    
    def get_scalers(self):
        scalers = list(set([classifiers_setup_dict[clf_name]['scaler'] for clf_name in classifiers_setup_dict.keys()]))
        return scalers      

    def make_log(self, imbalance_approach_class, imbalance_approach, 
                 imbalance_approach_hyperparameters, learning_algorithm, scaler, 
                 learning_algorithm_hyperparameters, test_balanced_accuracy, 
                 test_precision, test_recall, test_f1_macro, test_roc_auc, 
                 test_confusion_matrix, fit_time):
        
        summary = {}
        summary['experiment_name'] = self.name
        summary['metadata'] = self.metadata
        summary['train_size'] = self.train_size

        if type(imbalance_approach)==type(None):
            summary['imbalance_approach'] =  '' 
            summary['imbalance_approach_hyperparams'] = {}
            learning_algorithm_name = str(learning_algorithm.__class__)
        else:
            summary['imbalance_approach'] =  imbalance_approach.__name__ 
            summary['imbalance_approach_hyperparams'] = imbalance_approach_hyperparameters
            if(imbalance_approach_class=='cost-sensitive learning'):
                learning_algorithm_name = str(learning_algorithm.get_classifier_contructor())
            else:
                learning_algorithm_name = str(learning_algorithm.__class__)
        
        #Special treatment to metaalgorithm
        summary['learning_algorithm'] = learning_algorithm_name
        summary['learning_algorithm_hyperparams'] = learning_algorithm_hyperparameters
        summary['test_balanced_accuracy'] = round(test_balanced_accuracy, 4)
        summary['test_precision'] = round(test_precision, 4)
        summary['test_recall'] = round(test_recall, 4)
        summary['test_f1_macro'] = round(test_f1_macro, 4)
        summary['test_roc_auc'] = round(test_roc_auc, 4)
        summary['test_confusion_matrix'] = test_confusion_matrix
        summary['fit_time'] = round(fit_time, 6)

        return self.check_log(summary)

    def start_experiment(self, n_jobs=None):
        logs=[]
        params_grid = self.create_params_grid()
        for selection_grid in tqdm.tqdm(params_grid):
            best_params = self.select_hyperparameters(selection_grid)
            params_and_test_metrics = self.predict_test(best_params)
            logs.append(self.make_log(*params_and_test_metrics))
        return logs

    def run(self):
        self.split_dataset()
        self.scale_dataset(self.get_scalers())
        logs = self.start_experiment()
        self.save_experiment_logs(logs)

    def __init__(self, df, target, experiment_name, log_directory,  train_size=0.8, 
                 n_splits_cv=5, scoring=['balanced_accuracy', 'f1_macro', 'roc_auc', 
                 'precision', 'recall'], maximize_metric='balanced_accuracy',
                 metadata=None):
        """
        Initialize experiment object
        -----------------------------------------------------------------------------------------
        Input:
        df: Pandas DataFrame containing all variables (including target)
        target: Name of the dependent variable on df (str)
        experiment_name: Name of the experiment, which will be used to log results (str)
        log_directory: Directory where experiment will be saved (str)
        train_size: Percentage of dataset that will be used for training (float)
        n_splits_cv: Number of splits used to perform cross validation (int >1)
        scoring: Model performance metrics (e.g. 'accuracy', 'precision', ...) (list of str)
        maximize_metric: Model performance metric to be optimized during training (str)
        metadata: (Optional) Dictionary containing parameters about the dataset. If provided,
        the metadata will be passed as columns on experiment log file (dictionary)
        -----------------------------------------------------------------------------------------
        Output:
        None (this is a class constructor)
        """
        #Tests are inside setters
        self.set_experiment_name(experiment_name)
        self.set_log_directory(log_directory)
        self.config_log()
        self.set_XY(df, target)
        self.set_train_size(train_size)
        self.set_n_splits_cv(n_splits_cv)
        self.set_scoring(scoring)
        self.set_maximization_metric(maximize_metric)
        self.set_metadata(metadata)


        

       
        