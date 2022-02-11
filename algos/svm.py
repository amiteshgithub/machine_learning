import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from sklearn.svm import SVC

from algos.classifier_base import ClassificationBase
from algos.classifier_base import do_eda

from constants import TELCO_CHURN_DATA


class MlSVMClassification(ClassificationBase):
    def __init__(self, target_variable):
        self.model = SVC()
        self.param_grid = {
            'kernel': ['linear', 'rbf'],
            'C': [0.1, 1, 10],
            'probability': [True]  # Need this to be True in order to get probability and roc curve
        }
        self.plt_folder = 'state_vector_machine'
        super(MlSVMClassification, self).__init__(target_variable)


def state_vector_machine(data_file, target_variable):
    # create object with target variable name
    ml = MlSVMClassification(target_variable)

    # Do EDA
    df_with_dummies = do_eda(ml, data_file)

    # Check cross validation scores
    # ml.check_cross_validation_scores(df_with_dummies, cv=5, scoring='r2')
    ml.check_cross_validation_scores(df_with_dummies, cv=5)

    # predict on split data
    ml.plot_roc_curve_and_feature_importance(df_with_dummies)


if __name__ == '__main__':
    state_vector_machine(TELCO_CHURN_DATA, 'Churn') # with scaler and removal of high correlated features
