"""
Implementation of a complete decision tree algorithm capable of both 
classification and regression on both discrete and continuous data.
Input data set: The training pandas dataframe or the X_train and the training labels or y_train.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal

def check_ifreal(y: pd.Series, real_distinct_threshold: int = 6) -> bool:
    """
    Function to check if the given series has real or discrete values
    Returns True if the series has real (continuous) values, False otherwise (discrete).
    Examples:
    >>> check_ifreal(pd.Series(["A", "B", "C"]))
    False
    >>> check_ifreal(pd.Series([1,2,3,1,2,3,1,2,3]))
    False
    >>> check_ifreal(pd.Series([1.1, 2.2, 3.4, 5.6, 1, 0, 7]))
    True
    """

   # Check if the Series is of categorical type, boolean, or string
    if isinstance(y.dtype, pd.CategoricalDtype) or isinstance(y.dtype, bool) or isinstance(y.dtype, str):
        return False
    # Check if the series is of float type
    if pd.api.types.is_float_dtype(y):
        return True
    # Check if the series is of integer type and has enough unique values
    if pd.api.types.is_integer_dtype(y):
        return len(np.unique(y)) >= real_distinct_threshold
    return False


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    entropy = -sum(p_i * log2(p_i))
    Examples:
    >>> entropy(pd.Series([0,0,0,1,1,1]))
    1.0
    >>> entropy(pd.Series([0,0,0,0,0,0]))
    -0.0
    >>> entropy(pd.Series([1,1,1,1,1,1]))
    -0.0
    """
    
    value_counts = Y.value_counts()
    prob = value_counts / Y.size
    entropy = -np.sum(prob * np.log2(prob+1e-10))
    return np.round(entropy,1)

def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    gini_index = 1 - sum(p_i^2)
    Examples:
    >>> gini_index(pd.Series([0,0,0,1,1,1]))
    0.5
    >>> gini_index(pd.Series([0,0,0,0,0,0]))
    0.0
    >>> gini_index(pd.Series([1,1,1,1,1,1]))
    0.0
    """
    value_counts = Y.value_counts()
    probs = value_counts / Y.size
    gini_index_value = 1 - np.sum(probs ** 2)

    return gini_index_value


def mse(Y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    mse = sum((y_i - y)^2) / n
    Examples:
    >>> mse(pd.Series([0,1,2,3,4]))
    2.0
    >>> mse(pd.Series([0,0,0,0,0,0]))
    0.0
    >>> mse(pd.Series([1,5,3,7,9]))
    8.0
    """
    Y_mean = Y.mean()
    mse = np.sum((Y - Y_mean) ** 2) / Y.size
    return mse

def check_criteria(Y:pd.Series, criterion: str) -> str:
    """
    Function to check if the criterion is valid
    Returns the criterion
    """

    if criterion == "information_gain":
        if check_ifreal(Y):
            this_criteria = 'mse'
        else:
            this_criteria = 'entropy'
    if criterion == "gini_index":
        this_criteria = 'gini_index'
    
    criterion_funcs_map = {
        'entropy': entropy,
        'gini_index': gini_index,
        'mse': mse
    }
    criterion_func = criterion_funcs_map[this_criteria]
    return this_criteria, criterion_func



def opt_threshold(Y: pd.Series, attr: pd.Series, criterion) -> float:
    """
    Function to find the optimal threshold for a real feature
    Returns the threshold value
    """

    this_criteria, criterion_func = check_criteria(Y, criterion)

    sorted_attr = attr.sort_values()
    # Find the split points by taking the average of consecutive values (midpoints)
    if sorted_attr.size == 1:
        return None
    elif sorted_attr.size == 2:
        return (sorted_attr.sum()) / 2
    split_points = (sorted_attr[:-1] + sorted_attr[1:]) / 2

    best_threshold = None
    opt_gain = -np.inf

    for threshold in split_points:
        Y_left = Y[attr <= threshold]
        Y_right = Y[attr > threshold]

        if Y_left.empty or Y_right.empty:
            continue

        total_criterion = Y_left.size / Y.size * criterion_func(Y_left) + Y_right.size / Y.size * criterion_func(Y_right)

        information_gain_value = criterion_func(Y) - total_criterion

        if information_gain_value > opt_gain:
            best_threshold = threshold
            opt_gain = information_gain_value

    return best_threshold



def information_gain(Y: pd.Series, attribute: pd.Series, criterion = None) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)

    information_gain = criterion(Y) - sum((Y_i.size / Y.size) * criterion(Y_i))

    Raises:
    - ValueError: If the criterion is not one of 'entropy', 'gini', or 'mse'.
    """

    this_criteria, criterion_func = check_criteria(Y, criterion)

    # If the attribute is real, find the split points and calculate the information gain for each split point
    if check_ifreal(attribute):
        threshold = opt_threshold(Y, attribute, criterion)
        if threshold is None:
            return 0  # No valid threshold found
        Y_left = Y[attribute <= threshold]
        Y_right = Y[attribute > threshold]

        information_gain_value = criterion_func(Y) - (Y_left.size / Y.size * criterion_func(Y_left) + Y_right.size / Y.size * criterion_func(Y_right))
        
        return information_gain_value
    
    # If the attribute is discrete, calculate the information gain for each unique value of the attribute
    total_criterion = 0

    for value in attribute.unique():
        Y_i = Y[attribute == value]
        total_criterion += (Y_i.size / Y.size) * criterion_func(Y_i)

    information_gain_value = criterion_func(Y) - total_criterion

    return information_gain_value




def opt_split_attribute(X: pd.DataFrame, y: pd.Series, features: pd.Series, criterion: str) -> str:
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """

    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).

    best_feature = None
    opt_gain = -np.inf

    for feature in features:
        gain = information_gain(y, X[feature], criterion)

        if gain > opt_gain:
            best_feature = feature
            opt_gain = gain

    return best_feature


def split_data_discrete(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Funtion to split the data according to an attribute.
    """
    
    X_left = X[X[attribute] == value]
    X_right = X[X[attribute] != value]

    y_left = y.loc[X_left.index]
    y_right = y.loc[X_right.index]

    return X_left, y_left, X_right, y_right


def split_data_real(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Funtion to split the data according to an attribute.
    """
    
    X_left = X[X[attribute] <= value]
    X_right = X[X[attribute] > value]

    y_left = y.loc[X_left.index]
    y_right = y.loc[X_right.index]

    return X_left, y_left, X_right, y_right

def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """

    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.
    
    if check_ifreal(X[attribute]):
        X_left = X[X[attribute] <= value]
        X_right = X[X[attribute] > value]

    else:
        X_left = X[X[attribute] == value]
        X_right = X[X[attribute] != value]

    y_left = y.loc[X_left.index]
    y_right = y.loc[X_right.index]

    return X_left, y_left, X_right, y_right

@dataclass
class Node:
    # The attribute to split upon for non-leaf nodes, None for leaf nodes, and the output value for leaf nodes, gain is the information gain of the split, is_leaf is a boolean value to check if the node is a leaf node, value is the threshold value for real attributes, left and right are the left and right child nodes.
    def __init__(self, attribute=None, value=None, left=None, right=None, is_leaf=False, output=None, gain=0):
        self.attribute = attribute
        self.value = value
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.output = output
        self.gain = gain
    
    # Function to check if the node is a leaf node
    def check_leaf(self):
        return self.is_leaf



class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root_node = None


    def fit(self, X: pd.DataFrame, y: pd.Series, depth=0) -> None:
        """
        Function to train and construct the decision tree
        """

        # If you wish your code can have cases for different types of input and output data (discrete, real)
        # Use the functions from utils.py to find the optimal attribute to split upon and then construct the tree accordingly.
        # You may(according to your implemetation) need to call functions recursively to construct the tree.

        # If the depth exceeds max_depth or all the target values are the same, create a leaf node
        def build(X: pd.DataFrame, y: pd.Series, depth: int) -> Node:  
            #checking if maximum depth is reached or all the target values are same
            if depth >= self.max_depth or y.nunique() == 1:
                # If the target values are real, return the mean of the target values
                if check_ifreal(y):
                    return Node(is_leaf=True, output=np.round(y.mean(),4))
                # If the target values are discrete, return the mode of the target values
                else:
                    return Node(is_leaf=True, output=y.mode()[0])
            
            # Find the best attribute to split upon
            best_attribute = opt_split_attribute(X, y, X.columns, self.criterion)

            # If no good split is found, create a leaf node
            if best_attribute is None:
                if check_ifreal(y):
                    return Node(is_leaf=True, output=np.round(y.mean(),4))
                else:
                    return Node(is_leaf=True, output=y.mode()[0])

            if check_ifreal(X[best_attribute]):
                opt_val = opt_threshold(y, X[best_attribute], self.criterion)
            else:
                opt_val = X[best_attribute].mode()[0]

            # Split the data based on the best attribute and value
            X_left, y_left, X_right, y_right = split_data(X, y, best_attribute, opt_val)

            # If a valid split is not possible, create a leaf node
            if X_left.empty or X_right.empty:
                if check_ifreal(y):
                    return Node(is_leaf=True, output=np.round(y.mean(),4))
                else:
                    return Node(is_leaf=True, output=y.mode()[0])
                
            best_gain = information_gain(y, X[best_attribute], self.criterion)

            # Recursively build the left and right subtrees
            left = build(X_left, y_left, depth + 1)
            right = build(X_right, y_right, depth + 1)

            return Node(attribute=best_attribute, value=opt_val, left=left, right=right, gain=best_gain)

        # Start building the tree
        self.root_node = build(X, y, depth)
    

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """

        # Traverse the tree you constructed to return the predicted values for the given test inputs.

        def predict_single(x: pd.Series) -> float:
            """
            Function to predict the output for a single row of input
            """
            current_node = self.root_node
            while not current_node.check_leaf():
                if check_ifreal(x[current_node.attribute]):
                    if x[current_node.attribute] <= current_node.value:
                        current_node = current_node.left
                    else:
                        current_node = current_node.right
                else:
                    if x[current_node.attribute] == current_node.value:
                        current_node = current_node.left
                    else:
                        current_node = current_node.right
            return current_node.output
            
        return pd.Series([predict_single(x) for _, x in X.iterrows()])
    
    def print_tree(self) -> str:
        def print_node(node: Node, indent: str = '') -> str:
            if node.is_leaf:
                return f'Class {node.output}\n'
            
            # Format the non-leaf node
            result = f'?(attribute {node.attribute} <= {node.value:.2f})\n'
            result += f'{indent}Y: {print_node(node.left, indent + "    ")}'
            result += f'{indent}N: {print_node(node.right, indent + "    ")}'
            
            return result

        if not self.root_node:
            return "Tree not trained yet"
        else:
            return print_node(self.root_node)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
