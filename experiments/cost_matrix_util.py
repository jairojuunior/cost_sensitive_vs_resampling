from  sklearn.utils.class_weight import compute_class_weight
import numpy as np


def create_binary_balanced_cost_matrix(y): 
    classes=np.unique(y) 
    costs = compute_class_weight(class_weight='balanced', classes=classes, y=y) 
    C = [[0, round(costs[0],4)], [round(costs[1],4),0]] 
    return C 

def create_binary_manual_cost_matrix(manual_cost_tuple):
    C = [[0, round(manual_cost_tuple[0],4)], [round(manual_cost_tuple[1],4),0]] 
    return C

def create_binary_cost_matrix(cost, y=None):
    if type(cost)==str:
        if cost=='balanced':
            assert type(y)!=type(None), "Whem using 'balanced' cost matrix approach, y can not be None"
            return create_binary_balanced_cost_matrix(y)
        else:
            raise(cost+" cost matrix approach is not supported")
    elif type(cost)==tuple or type(cost)==list:
        if len(cost)==2:
            return create_binary_manual_cost_matrix(cost)
        else:
            raise("This implementation expects an array of lenght 2 as input, where every element is the misclassification cost of a class")
    else:
        raise("This implementation expects a tuple or array with length 2 or a string 'balanced'")

if __name__=='__main__':
    assert create_binary_cost_matrix([1,2])==[[0,1],[2,0]]
    assert create_binary_cost_matrix((1,2))==[[0,1],[2,0]]
    assert create_binary_cost_matrix('balanced', y=[0,0,0,0,0,0,0,1,1,1])==[[0, 0.7143], [1.6667, 0]]
    print("Passed!")
