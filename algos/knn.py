import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from sklearn.neighbors import KNeighborsClassifier

from algos.classifier_base import do_eda

from algos.classifier_base import ClassificationBase
from constants import TELCO_CHURN_DATA


class MlKNNClassification(ClassificationBase):
    def __init__(self, target_variable):
        self.model = KNeighborsClassifier()
        self.param_grid = {
            'n_neighbors': [1, 3, 5, 10, 15, 20],
            'metric': ['euclidean', 'manhattan']
        }
        self.plt_folder = 'knn'
        super(MlKNNClassification, self).__init__(target_variable)

def knn(data_file, target_variable):
    # create object with target variable name
    ml = MlKNNClassification(target_variable)

    # Do EDA
    df_with_dummies = do_eda(ml, data_file, scaling_type='standard')

    # Check cross validation scores
    # ml.check_cross_validation_scores(df_with_dummies, cv=5, scoring='r2')
    ml.check_cross_validation_scores(df_with_dummies, cv=5)

    # predict on split data
    ml.plot_roc_curve_and_feature_importance(df_with_dummies)

if __name__ == '__main__':
    knn(TELCO_CHURN_DATA, 'Churn') # with scaler and removal of high correlated features
