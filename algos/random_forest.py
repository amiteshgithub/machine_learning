import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from sklearn.ensemble import RandomForestClassifier

from algos.classifier_base import do_eda

from algos.classifier_base import ClassificationBase
from constants import TELCO_CHURN_DATA


class MlRandomTreeClassification(ClassificationBase):
    def __init__(self, target_variable):
        self.model = RandomForestClassifier()
        self.param_grid = {
            'n_estimators': [5, 50, 250],
            'max_depth': [2, 4, 8, 16, 32, None]
        }
        self.plt_folder = 'random_forest'
        super(MlRandomTreeClassification, self).__init__(target_variable)


def random_forest(data_file, target_variable):
    # create object with target variable name
    ml = MlRandomTreeClassification(target_variable)

    # Do EDA
    df_with_dummies = do_eda(ml, data_file)

    # Check cross validation scores
    # ml.check_cross_validation_scores(df_with_dummies, cv=5, scoring='r2')
    ml.check_cross_validation_scores(df_with_dummies, cv=5)

    # predict on split data
    ml.plot_roc_curve_and_feature_importance(df_with_dummies)


if __name__ == '__main__':
    random_forest(TELCO_CHURN_DATA, 'Churn') # with scaler and removal of high correlated features
