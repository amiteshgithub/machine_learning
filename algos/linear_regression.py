import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, r2_score

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from algos.ml_base import MlBase
from constants import HOUSING_DATA


class MlLinearRegression(MlBase):
    def __init__(self, target_variable):
        self.plt_folder = 'liner_regression'
        super(MlLinearRegression, self).__init__(target_variable)

    # statsmodel related
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
            X_1 = sm.add_constant(X_1)
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

    # statsmodel related
    def fit_model_and_plot_residual(self, df):
        df_copied = df.copy()
        # Model with selected continous + categorical features on full data set
        Y = df_copied[self.target_variable]
        X_with_dummies = df_copied.drop([self.target_variable], axis=1)  # Continuous + Categorical
        X_with_dummies = sm.add_constant(X_with_dummies)
        model = sm.OLS(Y, X_with_dummies)
        results = model.fit()
        print(results.summary())
        self.plot_residual(results)

    # statsmodel related
    def plot_residual(self, results):
        # Use statsmodels to plot the residuals vs predictions
        fig = plt.figure(figsize=(12, 8))
        plt.scatter(results.predict(), results.resid);
        plt.title("Residual plot for OLS Model")
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.savefig(
            '%s/plots/%s.png' % (
                os.path.join(os.path.dirname(__file__), os.pardir),
                'Residual'
            )
        )
        plt.close(fig)

    # scilearn related
    # Todo: should we add constant in X's below
    def plot_roc_curve_and_feature_importance(self, df, regression=None):
        print('#' * 50)
        def print_results_for_each_hyperparam(results):
            print('BEST PARAMS: {}\n'.format(results.best_params_))

            means = results.cv_results_['mean_test_score']
            stds = results.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, results.cv_results_['params']):
                print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))


        # Split data 80/20
        X_train, X_test, Y_train, Y_test = self.split_data(df, test_size=0.2, random_state=42)
        X_train = sm.add_constant(X_train)
        X_test = sm.add_constant(X_test)

        if (regression in ['ridge', 'lasso'] ):
            # find optimal alpha with grid search
            if (regression == 'ridge'):
                # By default is uses default hyper params (check using help(Ridge())
                model = Ridge()
            else:
                # By default is uses default hyper params (check using help(Lasso())
                model = Lasso()

            alpha = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
            param_grid = dict(alpha=alpha)
            cv = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                scoring='r2',
                verbose=1,
                n_jobs=-1,
                cv=5
            )

            grid_result = cv.fit(X_train, Y_train)
            # Print results for each param (alpha is the hyper param here)
            print_results_for_each_hyperparam(cv)

            # If we ever want to see results for each cross validation within gridsearchcv use:
            # print(cv.cv_results_)

            print('Best Score: ', grid_result.best_score_)
            print('Best Params: ', grid_result.best_params_)
            model = cv.best_estimator_
            model.fit(X_train, Y_train)
        elif (regression == 'linear_regression'):
            model = LinearRegression()
            model.fit(X_train, Y_train)
        elif (regression == 'sm_ols'):
            model = sm.OLS(Y_train, X_train)
            results = model.fit()
            print('Results Summary with training data')
            print(results.summary())
        else:
            raise Exception(
                'regression model passed: %s, Allowed: ["ridge", "linear_regression", "sm_ols"]'
            )

        if (regression == 'sm_ols'):
            # Predict SalePrice using test data
            Y_predict = results.predict(X_test)

            # R2 of test set using this model
            print('r2 score on test data: ', r2_score(Y_test, Y_predict))

            # Plot feature importance
            self.plot_feature_importance(X_train, results)
        else:
            print('model score on training data with bestparams: ', model.score(X_train, Y_train))

            # Predict SalePrice using test data
            Y_predict = model.predict(X_test)

            # R2 of test set using this model
            print('r2 score on test data: ', r2_score(Y_test, Y_predict))

            # Plot feature importance
            self.plot_feature_importance_scilearn(X_train, model)

        # Below scores can only be used for classification not for regression
        # Like we can use it in LogisticRegression
        # if (regression != 'sm_ols'):
        #     # Get accuracy scores with Test data (we should do it first with validation data)
        #     accuracy = round(accuracy_score(Y_test, Y_predict), 3)
        #     precision = round(precision_score(Y_test, Y_predict), 3)
        #     recall = round(recall_score(Y_test, Y_predict), 3)
        #     print('params: {} -- A: {} / P: {} / R: {}'.format(
        #         grid_result.best_params_,accuracy, precision, recall)
        #     )

        # Plot
        fig = plt.figure(figsize=(8, 8))
        plt.scatter(Y_test, Y_predict)
        plt.grid(True)
        plt.plot([0, Y_predict.max()], [0, Y_predict.max()], color='red')
        plt.title('Predicted vs. Actual Sale Price')
        plt.ylabel('Sale Price Predicted')
        plt.xlabel('Sale Price Actual')
        plt.savefig(
            '%s/plots/%s.png' % (
                os.path.join(os.path.dirname(__file__), os.pardir),
                'prediction_plot')
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
            '%s/plots/%s.png' % (
                os.path.join(os.path.dirname(__file__), os.pardir),
                'feature_importance')
        )
        plt.close(featfig)

    def plot_feature_importance_scilearn(self, X, model):
        feature_importance = abs(model.coef_)

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
            '%s/plots/%s.png' % (
                os.path.join(os.path.dirname(__file__), os.pardir),
                'feature_importance')
        )
        plt.close(featfig)

    def check_cross_validation_scores(self, df, cv=5, scoring=None, regression=None):
        print('#'*50)
        if (regression == 'ridge'):
            lf = Ridge()
        elif (regression == 'lasso'):
            lf = Lasso()
        else:
            lf = LinearRegression()

        X = self.drop_columns(df, columns=[self.target_variable])
        X = sm.add_constant(X)
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
#################End of class MlLinearRegression########


# Does EDA (Do these things one by one on your new data)
def do_eda(ml):
    ##############EDA################
    # Read csv (send csv file variable name)
    ml.set_dafa_frame_from_csv(HOUSING_DATA)
    print(ml.df.head())

    # Drop irrelevant column (ID)
    ml.df = ml.drop_columns(ml.df, columns=['Id'])

    # Check info to see data types
    print(ml.get_df_info(ml.df))

    # Extract continuous features df
    ml.df_continous = ml.get_continuous_features(ml.df)

    # Fill missing values in continuous df with mean
    ml.fill_missing_values_in_continuous(ml.df_continous)
    ml.df_continous.head()

    # Extract categorical features df
    ml.df_categorical = ml.get_categorical_features(ml.df)

    # Get stats and plot for categorical features
    ml.get_categorical_feature_stats(ml.df_categorical)

    # Fill missing values in categorical features df with either highest frequency value of 'UNKNOWN'
    ml.fill_missing_values_in_categorical(ml.df_categorical, level=0.1)

    # Drop single valued columns
    single_valued_columns = ml.get_n_valued_categorical_columns(ml.df_categorical, n=1)
    ml.df_categorical.drop(columns=single_valued_columns, inplace=True)

    # Transform binary columns (In place)
    ml.transform_binary_columns(ml.df_categorical)

    # Transform multi valued columns
    ml_cat_with_dummies = ml.transform_multi_valued_columns(ml.df_categorical)
    print('df_cat_with_dummies:')
    print(ml_cat_with_dummies.head())

    # Scale continuous data (drop target variable)
    ml.df_continous_scaled = ml.get_scaled_df(
        ml.drop_columns(ml.df_continous, columns=[ml.target_variable])
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

    # get df after Backward elimination (join target variable column back to it before backward
    # elimination)
    df_with_dummies = df_with_dummies.join(ml.df[ml.target_variable])
    df_with_dummies = ml.backward_eliminate(df_with_dummies)

    # Plot heatmaps (after removing high correlation columns and backward elimination)
    ml.plot_heatmap_continuous(df_with_dummies)
    ml.plot_heatmap_all(df_with_dummies)

    ##############EDA Done################

    return df_with_dummies

# statsmodel related
def orginal_regression_submitted():
    ##############EDA################
    # create object with target variable name
    ml = MlLinearRegression('SalePrice')

    # Read csv (send csv file variable name)
    ml.set_dafa_frame_from_csv(HOUSING_DATA)
    print(ml.df.head())

    # Drop irrelevant column (ID)
    ml.df = ml.drop_columns(ml.df, columns=['Id'])

    # Check info to see data types
    print(ml.get_df_info(ml.df))

    # Extract continuous features df
    ml.df_continous = ml.get_continuous_features(ml.df)

    # Fill missing values in continuous df with mean
    ml.fill_missing_values_in_continuous(ml.df_continous)
    ml.df_continous.head()

    # Extract categorical features df
    ml.df_categorical = ml.get_categorical_features(ml.df)

    # Get stats and plot for categorical features
    ml.get_categorical_feature_stats(ml.df_categorical)

    # Fill missing values in categorical features df with either highest frequency value of 'UNKNOWN'
    ml.fill_missing_values_in_categorical(ml.df_categorical, level=0.1)

    # Get dummies
    ml_dummies = ml.get_dummy_variables(ml.df_categorical)
    print(ml_dummies.head())

    # join categorical dummies and continuous
    df_with_dummies = ml.df_continous.join(ml_dummies)

    # get df after Backward elimination
    df_with_dummies = ml.backward_eliminate(df_with_dummies)

    # Plot heatmaps
    ml.plot_heatmap_continuous(df_with_dummies)
    ml.plot_heatmap_all(df_with_dummies)

    ##############EDA Done################

    #fit model and plot residual
    ml.fit_model_and_plot_residual(df_with_dummies)

    # predict on split data
    ml.plot_roc_curve_and_feature_importance(df_with_dummies, regression='sm_ols')


def linear_regression(regression=None):
    # create object with target variable name
    ml = MlLinearRegression('SalePrice')

    # Do EDA
    df_with_dummies = do_eda(ml)

    if (regression!='sm_ols'):
        # Check cross validation scores
        ml.check_cross_validation_scores(df_with_dummies, cv=5, scoring='r2', regression=regression)

    # predict on split data
    ml.plot_roc_curve_and_feature_importance(df_with_dummies, regression=regression)

if __name__ == '__main__':
    # orginal_regression_submitted()

    ###########################
    # linear_regression(regression='sm_ols') # with scaler and removal of high correlated features
    # linear_regression(regression='linear_regression')
    # linear_regression(regression='lasso')
    linear_regression(regression='ridge')