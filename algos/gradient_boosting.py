import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from sklearn.ensemble import GradientBoostingClassifier

from algos.classifier_base import do_eda

from algos.classifier_base import ClassificationBase
from constants import TELCO_CHURN_DATA


class MlGradientBoostingClassification(ClassificationBase):
    def __init__(self, target_variable):
        self.model = GradientBoostingClassifier()
        self.param_grid = {
            'n_estimators': [500, 1000, 2000],
            'learning_rate': [.001, 0.01, .1],
            'max_depth': [1, 2, 4],
            'subsample': [.5, .75, 1],
            'random_state': [1]
        }
        self.plt_folder = 'gradient_boost'
        super(MlGradientBoostingClassification, self).__init__(target_variable)

def gradient_boost(data_file, target_variable):
    # create object with target variable name
    ml = MlGradientBoostingClassification(target_variable)

    # Do EDA
    df_with_dummies = do_eda(ml, data_file)

    # Check cross validation scores
    # ml.check_cross_validation_scores(df_with_dummies, cv=5, scoring='r2')
    ml.check_cross_validation_scores(df_with_dummies, cv=5)

    # predict on split data
    ml.plot_roc_curve_and_feature_importance(df_with_dummies)

if __name__ == '__main__':
    gradient_boost(TELCO_CHURN_DATA, 'Churn') # with scaler and removal of high correlated features
