import os
import sys

from sklearn.linear_model import LogisticRegression

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from algos.classifier_base import ClassificationBase
from constants import TELCO_CHURN_DATA

from algos.classifier_base import do_eda


class MlLogisticRegression(ClassificationBase):
    def __init__(self, target_variable):
        self.model = LogisticRegression()
        self.param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'penalty': ['l1', 'l2'],
            # 'solver': ['newton-cg', 'liblinear']
            'solver': ['liblinear']

        }
        self.plt_folder = 'logistic_regression'
        super(MlLogisticRegression, self).__init__(target_variable)


def logistic_regression(data_file, target_variable):
    # create object with target variable name
    ml = MlLogisticRegression(target_variable)

    # Do EDA
    df_with_dummies = do_eda(ml, data_file)

    # Check cross validation scores
    # ml.check_cross_validation_scores(df_with_dummies, cv=5, scoring='r2')
    ml.check_cross_validation_scores(df_with_dummies, cv=5)

    # predict on split data
    ml.plot_roc_curve_and_feature_importance(df_with_dummies)


if __name__ == '__main__':
    logistic_regression(TELCO_CHURN_DATA, 'Churn') # with scaler and removal of high correlated features
