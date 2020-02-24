"""
This file contains all the non deep learning models
"""
import pickle
import sys

import numpy
import sklearn
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from speechemotion.mlcode.model_base_class import Model


class MLModel(Model):
    """
    This class is parent class for all Non Deep learning models
    """

    def __init__(self, **params):
        super(MLModel, self).__init__(**params)

    def save_model(self, save_path):
        pickle.dump(self.model, open(save_path, "wb"))

    def _load_model(self, to_load: str):
        try:
            self.model = pickle.load(open(to_load, "rb"))
        except:
            sys.stderr.write("Invalid saved file provided")
            sys.exit(-1)

    def fit(self, x_train, y_train, validation_data=None):
        self.model.fit(x_train, y_train)
        self.trained = True

    def predict(self, samples):
        if not self.trained:
            sys.stderr.write(
                "Model should be trained or loaded before doing predict\n")
            sys.exit(-1)
        return self.model.predict(samples)



class SVM(MLModel):
    """
    SVM implements use of SVM for speech emotion recognition
    """

    def __init__(self, **params):
        params['name'] = 'SVM'
        super(SVM, self).__init__(**params)
        self.model = LinearSVC(multi_class='crammer_singer')


class RF(MLModel):
    """
    RF implements use of Random Forest for speech emotion recognition
    """

    def __init__(self, **params):
        params['name'] = 'Random Forest'
        super(RF, self).__init__(**params)
        self.model = RandomForestClassifier(n_estimators=30)


class NN(MLModel):
    """
    NN implements use of Neural networks for speech emotion recognition
    """

    def __init__(self, **params):
        params['name'] = 'Neural Network'
        super(NN, self).__init__(**params)
        self.model = MLPClassifier(activation='logistic', verbose=True,
                                   hidden_layer_sizes=(512,), batch_size=32)


class SKLearnModelAdapter(MLModel):
    """A adapter to transfer sk-learn model API to MLModel"""
    def __init__(self, sklearn_model, **params):
        if 'name' not in params:
            params['name'] = 'SKLearnModelAdapter'
        super(SKLearnModelAdapter, self).__init__(**params)
        self.params = params
        self.model = sklearn_model

    def clone_model(self):
        new_sklearn_model = sklearn.clone(self.model)
        return SKLearnModelAdapter(new_sklearn_model, **self.params)

    def log_parameters(self):
        model = self.model
        model_params_dict = {}
        if isinstance(model, linear_model.LogisticRegression):
            model_params_dict['lr_coef'] = model.coef_.tolist()
        if isinstance(model, linear_model.LogisticRegressionCV):
            model_params_dict['lr_C'] = list(model.C_)
        if isinstance(model, GridSearchCV):
            model_params_dict['grid_best_params'] = model.best_params_
        return model_params_dict
