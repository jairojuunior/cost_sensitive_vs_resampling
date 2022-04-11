from costcla.models import BayesMinimumRiskClassifier
from cost_matrix_util import create_binary_cost_matrix
import numpy as np

class BayesMinimumRisk(object):
    def __init__(self, classifier, cost):
        self.classifier = classifier
        self.cost = cost
            
    def __call__(self, **params): 
        return BayesMinimumRisk(self.get_classifier_contructor()(**params), self.cost)

    def get_params(self, deep=True):
        return {'classifier': self.classifier,
                'cost': self.cost}

    def set_params(self, deep=True, **params):
        if deep:
            return BayesMinimumRisk(**params)
        return self

    def get_classifier_contructor(self):
        return self.classifier.__class__

    def reshape_cost_matrix(self, size_y):
        tn, fp, fn, tp = np.reshape(self.cost_matrix, (4,))
        cost_example = [fp, fn, tp, tn]
        cost_matrix_reshaped = np.ones((size_y,4))*cost_example
        return cost_matrix_reshaped
        
    def fit(self, X, y):
        #Adaptation to balanced strategy
        if self.cost == 'balanced':
            self.cost_matrix = create_binary_cost_matrix(self.cost, y)
        else:
            self.cost_matrix = create_binary_cost_matrix(self.cost)

        base_model = self.classifier.fit(X, y)
        y_prob = base_model.predict_proba(X)
        bmr = BayesMinimumRiskClassifier()
        bmr.fit(y, y_prob) 
        self.set_fitted_model(base_model)
        self.set_bmr(bmr)
        return self

    def set_fitted_model(self, model):
        self.fitted_model = model

    def get_fitted_model(self):
        return self.fitted_model

    def set_bmr(self, bmr):
        self.bmr = bmr
    
    def get_bmr(self):
        return self.bmr

    def predict(self, X):
        model = self.get_fitted_model()
        bmr = self.get_bmr()
        C_mat = self.reshape_cost_matrix(len(X))
        y_prob = model.predict_proba(X)
        y_pred = bmr.predict(y_prob, C_mat) 
        return y_pred

    def predict_proba(self, X):
        model = self.get_fitted_model()
        return model.predict_proba(X) 