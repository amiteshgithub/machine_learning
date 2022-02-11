import os
import sys
from time import time
import joblib

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, r2_score, \
    roc_auc_score, roc_curve

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from algos.classifier_base import ClassificationBase, do_eda

from algos.decision_tree import decision_tree
from algos.logistic_regression import logistic_regression
from algos.xg_boost import xg_boost
from algos.random_forest import random_forest
from algos.multi_layer_perceptron import mlp
from algos.svm import state_vector_machine

from constants import TELCO_CHURN_DATA, TELCO_CHURN_DATA_TEST



from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC



class MlClassificationModelEstimator(ClassificationBase):
    def __init__(self, target_variable):
        self.target_variable = 'Churn'
        self.training_data_csv = TELCO_CHURN_DATA
        self.test_data_csv = TELCO_CHURN_DATA_TEST
        self.models = [
            {
                'name': 'xg_boost',
                'class': XGBClassifier,
                'params': {
                    'n_estimators': [5, 50, 250, 500],
                    'max_depth': [1, 3, 5, 7, 9],
                    'learning_rate': [0.01, 0.1, 1, 10, 100]
                },
                'plt_folder': os.path.join(os.path.dirname(__file__), os.pardir, 'plots', 'xg_boost')
            }
        ]
        self.plt_folder = 'test_dir'
        super(MlClassificationModelEstimator, self).__init__(self.target_variable)

    def explore_all_models(self):
        for model in self.models:
            self.model = model['class']()
            self.param_grid = model['params']
            self.plt_folder = model['plt_folder']

            # Do EDA
            df_with_dummies = do_eda(self, self.training_data_csv)

            # Check cross validation scores
            # ml.check_cross_validation_scores(df_with_dummies, cv=5, scoring='r2')
            self.check_cross_validation_scores(df_with_dummies, cv=5)

            # predict on split data
            self.plot_roc_curve_and_feature_importance(df_with_dummies)

    def plot_roc_curve_estimate_models(self, df_test):
        print('#' * 50)

        fig = plt.figure(figsize=(15, 10))
        # plot horizontal line
        plt.plot([0, 1], [0, 1], linestyle='--')

        def model_evaluation(pkl_file_model_list, X_test, Y_test):
            for pkl_file_model_dict in pkl_file_model_list:
                start = time()
                model = pkl_file_model_dict['model']
                name = pkl_file_model_dict['model_name']
                Y_predict = model.predict(X_test)
                end = time()
                accuracy = round(accuracy_score(Y_test, Y_predict), 3)
                precision = round(precision_score(Y_test, Y_predict), 3)
                recall = round(recall_score(Y_test, Y_predict), 3)

                # predict probabilities
                predictions = model.predict_proba(X_test)[:, 1]

                # calculate scores
                auc = roc_auc_score(Y_test, predictions)

                # calculate roc curves
                fpr, tpr, _ = roc_curve(Y_test, predictions)

                # print stats
                print(
                    'Results with Test Data for "{}" -- Accuracy: {} / Precision: {} / Recall: {} / AUC: {:.3f} /Latency: {}ms'.format(
                        name,
                        accuracy,
                        precision,
                        recall,
                        auc,
                        round((end - start) * 1000, 1))
                )

                # plot the roc curve for the model
                plt.plot(fpr, tpr, label='ROC curve %s (AUC = %0.3f)' % (name, auc))

            # axis labels
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            # show the legend
            plt.legend(loc='lower right')
            plt.savefig(
                '%s/%s.png' % (
                    os.path.join(os.path.dirname(__file__), os.pardir, 'plots'),
                    'roc_plot_all_algos')
            )
            plt.close(fig)


        # Retrieve X_test and Y_test from test data
        Y_test = df_test[self.target_variable]
        X_test = df_test.drop([self.target_variable], axis=1)

        # Get all .pkl files
        pkl_file_model_list = []
        path = os.path.join(os.path.dirname(__file__), os.pardir, 'plots')
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".pkl"):
                    print(os.path.join(root, file))
                    pkl_file_model_list.append(
                        {
                            'model': joblib.load(os.path.join(root, file)),
                            'file_name': os.path.basename(os.path.join(root, file)),
                            'abs_file_path': os.path.join(root, file),
                            'model_name': os.path.split(os.path.dirname(os.path.join(root, file)))[1]
                        }
                    )

        # Print all models details
        print('All models:')
        for pkl_file_model_dict in pkl_file_model_list:
            print(pkl_file_model_dict)

        # plot roc curve for each algo (on same plot and calculate stats for ech algo)
        model_evaluation(
            pkl_file_model_list,
            X_test,
            Y_test
        )
        print('#' * 50)


def evaluate_models(test_data_file, target_variable):
    # create object with target variable name
    ml = MlClassificationModelEstimator(target_variable)

    # Do EDA
    df_with_dummies = do_eda(ml, test_data_file, remove_outliers=False)

    # predict on split data
    ml.plot_roc_curve_estimate_models(df_with_dummies)

if __name__ == '__main__':
    # logistic_regression(data_file=TELCO_CHURN_DATA, target_variable='Churn') # with scaler and removal of high correlated features
    # decision_tree(data_file=TELCO_CHURN_DATA, target_variable='Churn')
    # random_forest(data_file=TELCO_CHURN_DATA, target_variable='Churn')
    # xg_boost(data_file=TELCO_CHURN_DATA, target_variable='Churn')
    # mlp(data_file=TELCO_CHURN_DATA, target_variable='Churn')
    # state_vector_machine(data_file=TELCO_CHURN_DATA, target_variable='Churn')

    evaluate_models(test_data_file=self.test_data_csv, target_variable=self.target_variable)
