import os
import sys
import time

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import joblib

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, r2_score, \
    roc_auc_score, roc_curve

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from algos.ml_base import MlBase


class ClassificationBase(MlBase):
    def __init__(self, target_variable):
        super(ClassificationBase, self).__init__(target_variable)

    # statsmodel related (Do not use it anymore)
    def backward_eliminate(self, df):
        # Backward Elimination for features (continuous + dummies)
        df_copied = df.copy()
        Y = df_copied[self.target_variable]  # Dependent Variable (Target)
        X = df_copied.drop([self.target_variable], axis=1)  # Independent variables
        cols = list(X.columns)
        print('Total number of features before backward elimination: ', len(X.columns))

        eleminated_features = []
        while (len(cols) > 0):
            p = []
            X_1 = X[cols]
            model = sm.OLS(Y, X_1).fit()
            p = pd.Series(model.pvalues.values[1:], index=cols)
            pmax = max(p)
            feature_with_p_max = p.idxmax()
            if (pmax > 0.05):
                cols.remove(feature_with_p_max)
                eleminated_features.append(feature_with_p_max)
            else:
                # Break out of look when p > 0.05 is no longer available for any feature
                break

        selected_features = cols
        print('Total number of features after backward elimination: ', len(selected_features))
        print('Selected features: ', selected_features)
        print('Eliminated features: ', eleminated_features)
        # Drop irerelevant features from df (p > 0.05)
        df_copied = df_copied[df_copied.columns[~df_copied.columns.isin(eleminated_features)]]

        return df_copied
    #################################

    def get_balanced_data(self, df):
        positive_labels = df[df[self.target_variable] == 1]
        num_positive_labels = positive_labels.shape[0]

        negative_labels = df[df[self.target_variable] == 0].sample(
            num_positive_labels)

        balanced_data = positive_labels.append(negative_labels)
        return balanced_data

    # scilearn related
    def plot_roc_curve_and_feature_importance(self, df):
        print('#' * 50)
        df_copied = df.copy()
        def print_results_for_each_hyperparam(results):
            print('BEST PARAMS: {}\n'.format(results.best_params_))

            means = results.cv_results_['mean_test_score']
            stds = results.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, results.cv_results_['params']):
                print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))

        # Get balanced data
        df_copied = self.get_balanced_data(df_copied)

        # Split data 80/20
        X_train, X_test, Y_train, Y_test = self.split_data(df_copied, test_size=0.2, random_state=42)

        cv = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grid,
            verbose=1,
            n_jobs=-1,
            cv=5 # Use cross validation while training to avoid over fitting
        )

        # fit model to get best hyper params
        cv.fit(X_train, Y_train.values.ravel())
        # Print results for each param (alpha is the hyper param here)
        print_results_for_each_hyperparam(cv)

        # If we ever want to see results for each cross validation within gridsearchcv use:
        # print(cv.cv_results_)

        print('Best Score: ', cv.best_score_)
        print('Best Params: ', cv.best_params_)
        model = cv.best_estimator_

        # Dump model with best estimator to compare later with other algos
        path = os.path.join(os.path.dirname(__file__), os.pardir, 'plots', self.plt_folder)
        joblib.dump(model, '%s/%s_model.pkl' % (path, self.plt_folder))

        model.fit(X_train, Y_train.values.ravel())
        print('model score on training data with best params: ', model.score(X_train, Y_train))

        # Predict target using test data
        Y_predict = model.predict(X_test)

        # predict probabilities
        # keep probabilities for the positive outcome only
        predictions = model.predict_proba(X_test)[:, 1]

        # Get accuracy scores with Test data (we should do it first with validation data)
        accuracy = round(accuracy_score(Y_test, Y_predict), 3)
        precision = round(precision_score(Y_test, Y_predict), 3)
        recall = round(recall_score(Y_test, Y_predict), 3)
        print('Hyper params: {} -- Accuracy: {} / Precision: {} / Recall: {}'.format(
            cv.best_params_, accuracy, precision, recall)
        )

        # Plot feature importance
        if (isinstance(model, LogisticRegression)):
            self.plot_feature_importance_scilearn(X_train, model)

        # calculate scores
        auc = roc_auc_score(Y_test, predictions)

        # calculate roc curves
        fpr, tpr, _ = roc_curve(Y_test, predictions)

        fig = plt.figure(figsize=(15, 10))
        # plot horizontal line
        plt.plot([0, 1], [0, 1], linestyle='--')
        # plot the roc curve for the model
        plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % auc)
        # axis labels
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        # show the legend
        plt.legend(loc='lower right')
        plt.savefig(
            '%s/%s.png' % (
                os.path.join(os.path.dirname(__file__), os.pardir, 'plots', self.plt_folder),
                'roc_plot')
        )
        plt.close(fig)
        print('#' * 50)

    # statsmodel related
    def plot_feature_importance(self, X, results):
        coefficients = []
        for column in X.columns:
            coefficients.append(abs(results.params[column]))

        feature_importance = np.array(coefficients)

        # feature_importance = 100.0 * (feature_importance / feature_importance.max())
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5

        featfig = plt.figure()
        featax = featfig.add_subplot(1, 1, 1)
        featax.barh(pos, feature_importance[sorted_idx], align='center')
        featax.set_yticks(pos)
        featax.set_yticklabels(np.array(X.columns)[sorted_idx], fontsize=8)

        plt.title('Feature Importance')
        plt.savefig(
            '%s/%s.png' % (
                os.path.join(os.path.dirname(__file__), os.pardir, 'plots', self.plt_folder),
                'feature_importance')
        )
        plt.close(featfig)

    def plot_feature_importance_scilearn(self, X, model):
        feature_importance = abs(model.coef_[0]) # This is only diff from one i have in linear reg.

        # feature_importance = 100.0 * (feature_importance / feature_importance.max())
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5

        featfig = plt.figure(figsize=(10, 15))
        featax = featfig.add_subplot(1, 1, 1)
        featax.barh(pos, feature_importance[sorted_idx], align='center')
        featax.set_yticks(pos)
        featax.set_yticklabels(np.array(X.columns)[sorted_idx], fontsize=8)

        plt.title('Feature Importance')
        plt.savefig(
            '%s/%s.png' % (
                os.path.join(os.path.dirname(__file__), os.pardir, 'plots', self.plt_folder),
                'feature_importance')
        )
        plt.close(featfig)

    def check_cross_validation_scores(self, df, cv=5, scoring=None):
        print('#'*50)

        lf = self.model

        X = self.drop_columns(df, columns=[self.target_variable])
        Y = self.df[self.target_variable]
        if (scoring):
            scores = cross_val_score(lf, X, Y.values.ravel(), cv=cv, scoring=scoring)
        else:
            scores = cross_val_score(lf, X, Y.values.ravel(), cv=cv)

        print('Scores: ', scores)
        print('CV Mean: ', np.mean(scores))
        print('STD: ', np.std(scores))
        print('\n')
        print('#' * 50)
#################End of class ClassificationBase########


# Does EDA (Do these things one by one on your new data)
def do_eda(ml, data_file, drop_first=False, remove_outliers=True, scaling_type='min_max'):
    ##############EDA################
    # Read csv (send csv file variable name)
    ml.set_dafa_frame_from_csv(data_file)
    print(ml.df.head())

    #
    # Drop irrelevant column (ID)
    #
    # Remove irrelavalent feature: customer ID (Dropping because all different values for each row)
    # Checked using:
    # Check number of different values of customerID
    # print('Number of different values for %s = %s' % (
    # 'customerID', telco_data_categorical['customerID'].value_counts().count()))
    # print('Value with highest frequency "%s"' %
    #       telco_data_categorical['customerID'].value_counts().index[0])
    #
    ml.df = ml.drop_columns(ml.df, columns=['customerID'])

    # Check info to see data types
    print(ml.get_df_info(ml.df))

    # Change data type for TotalCharges to float64
    # Got info from above print that totalcharges are object but should be numeric
    ml.df["TotalCharges"] = pd.to_numeric(ml.df['TotalCharges'], errors='coerce')

    # Encode target variable
    # Got info from above info print we should always convert our target in regression to 0 and 1s
    ml.df.Churn = ml.df.Churn.replace({'Yes': 1, 'No': 0})  # Replacing 'Yes' and 'No' with numerical values

#####All below same as linear nothing different ######
    # Drop empty columns
    ml.drop_empty_columns(ml.df)

    # Extract continuous features df
    ml.df_continous = ml.get_continuous_features(ml.df)

    # Drop columsn with more than 85% Nans
    ml.drop_high_nan_columns(ml.df_continous, 85)

    # Fill missing values in continuous df with mean
    ml.fill_missing_values_in_continuous(ml.df_continous)
    ml.df_continous.head()

    # Extract categorical features df
    ml.df_categorical = ml.get_categorical_features(ml.df)
    ml.df_categorical = ml.df_categorical.replace(' ', '_', regex=True)

    # Get stats and plot for categorical features
    ml.get_categorical_feature_stats(ml.df_categorical)

    # Fill missing values in categorical features df with either highest frequency value or 'UNKNOWN'
    ml.fill_missing_values_in_categorical(ml.df_categorical, level=0.1)

    # Drop single valued columns
    single_valued_columns = ml.get_n_valued_categorical_columns(ml.df_categorical, n=1)
    ml.df_categorical.drop(columns=single_valued_columns, inplace=True)

    # Transform binary columns (In place)
    ml.transform_binary_columns(ml.df_categorical)

    # Transform multi valued columns (Important : In decision trees we do not drop first)
    ml_cat_with_dummies = ml.transform_multi_valued_columns(ml.df_categorical, drop_first=drop_first)
    print('df_cat_with_dummies:')
    print(ml_cat_with_dummies.head())

    # Scale continuous data (drop target variable) (never scale target)
    ml.df_continous_scaled = ml.get_scaled_df(
        ml.drop_columns(ml.df_continous, columns=[ml.target_variable]),
        type=scaling_type
    )
    print('df_continous_scaled:')
    print(ml.df_continous_scaled.head())

    # join categorical dummies and continuous
    df_with_dummies = ml.df_continous_scaled.join(ml_cat_with_dummies)
    print('df_with_dummies:')
    print(df_with_dummies.head())

    # Drop columns with correlation > 0.8
    columns_to_drop = ml.get_columns_with_correlation_greater_than_x(
        df_with_dummies,
        max_correlation=0.8
    )
    df_with_dummies.drop(columns=columns_to_drop, axis=1, inplace=True)

    # done only for training data (not for test data)
    if (remove_outliers):
        # Remove outliers (remove rows where zscore > 3 for any column)
        df_with_dummies = ml.remove_outliers(df_with_dummies)

    #
    # get df after Backward elimination (join target variable column back to it) before backward
    # elimination)
    # --> Note upper comment states backward elimination (we are not using that anymore so ignore)
    #
    df_with_dummies = df_with_dummies.join(ml.df[ml.target_variable])
    print('Final DF:')
    print(df_with_dummies.head())
    # df_with_dummies = ml.backward_eliminate(df_with_dummies)

    # Plot heatmaps (after removing high correlation columns)
    ml.plot_heatmap_continuous(df_with_dummies)
    ml.plot_heatmap_all(df_with_dummies)

    ##############EDA Done################

    return df_with_dummies
