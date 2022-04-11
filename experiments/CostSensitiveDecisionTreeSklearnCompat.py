from costcla.models import CostSensitiveDecisionTreeClassifier
from cost_matrix_util import create_binary_cost_matrix
import numpy as np

class CostSensitiveDT(object):
    def __init__(self, cost, criterion='direct_cost', criterion_weight=False,
                 num_pct=100, max_features=None, max_depth=None, 
                 min_samples_split=2, min_samples_leaf=1, min_gain=0.001,
                 pruned=True):
        self.cost = cost
        self.criterion = criterion
        self.criterion_weight = criterion_weight
        self.num_pct = num_pct
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.pruned = pruned
        
            
    def __call__(self, **params): 
        return CostSensitiveDT(**params)

    def get_params(self, deep=True):
        return {'cost': self.cost,
                'criterion': self.criterion,
                'criterion_weight': self.criterion_weight,
                'num_pct': self.num_pct,
                'max_features': self.max_features,
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf,
                'min_gain': self.min_gain,
                'pruned': self.pruned}

    def set_params(self, deep=True, **params):
        if deep:
            return CostSensitiveDT(**params)
        return self

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
        C_mat_train = self.reshape_cost_matrix(len(X))
        params = self.get_params()
        model_params = {key: params[key] for key in params.keys() if key!='cost'}
        model = CostSensitiveDecisionTreeClassifier(**model_params)
        model.fit(X, y, C_mat_train) 
        self.set_fitted_model(model)
        return self

    def set_fitted_model(self, model):
        self.fitted_model = model

    def get_fitted_model(self):
        return self.fitted_model

    def predict(self, X):
        model = self.get_fitted_model()
        y_pred = model.predict(X) 
        return y_pred

    def predict_proba(self, X):
        model = self.get_fitted_model()
        return model.predict_proba(X) 