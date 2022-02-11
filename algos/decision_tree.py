import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from sklearn.tree import DecisionTreeClassifier

from algos.classifier_base import ClassificationBase
from algos.classifier_base import do_eda

from constants import TELCO_CHURN_DATA


class MlDecisionTreeClassification(ClassificationBase):
    def __init__(self, target_variable):
        self.model = DecisionTreeClassifier()
        self.param_grid = {
            'max_depth': [3, 5, 6, 10]
        }
        self.plt_folder = 'decision_tree'
        super(MlDecisionTreeClassification, self).__init__(target_variable)


def decision_tree(data_file, target_variable):
    # create object with target variable name
    ml = MlDecisionTreeClassification(target_variable)

    # Do EDA
    df_with_dummies = do_eda(ml, data_file)

    # Check cross validation scores
    # ml.check_cross_validation_scores(df_with_dummies, cv=5, scoring='r2')
    ml.check_cross_validation_scores(df_with_dummies, cv=5)

    # predict on split data
    ml.plot_roc_curve_and_feature_importance(df_with_dummies)


if __name__ == '__main__':
    decision_tree(TELCO_CHURN_DATA, 'Churn') # with scaler and removal of high correlated features
