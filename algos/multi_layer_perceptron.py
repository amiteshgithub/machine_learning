import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from sklearn.neural_network import MLPClassifier

from algos.classifier_base import do_eda

from algos.classifier_base import ClassificationBase
from constants import TELCO_CHURN_DATA


class MlMultiLayerPerceptronClassification(ClassificationBase):
    def __init__(self, target_variable):
        self.model = MLPClassifier()
        self.param_grid = {
            'hidden_layer_sizes': [(10,), (50,), (100,), (20, 20), (20, 30, 20)],
            'activation': ['identity', 'relu', 'tanh', 'logistic'],
            'learning_rate': ['constant', 'invscaling', 'adaptive']
        }
        self.plt_folder = 'multi_layer_perceptron'
        super(MlMultiLayerPerceptronClassification, self).__init__(target_variable)

def mlp(data_file, target_variable):
    # create object with target variable name
    ml = MlMultiLayerPerceptronClassification(target_variable)

    # Do EDA
    df_with_dummies = do_eda(ml, data_file)

    # Check cross validation scores
    # ml.check_cross_validation_scores(df_with_dummies, cv=5, scoring='r2')
    ml.check_cross_validation_scores(df_with_dummies, cv=5)

    # predict on split data
    ml.plot_roc_curve_and_feature_importance(df_with_dummies)

if __name__ == '__main__':
    mlp(TELCO_CHURN_DATA, 'Churn') # with scaler and removal of high correlated features
