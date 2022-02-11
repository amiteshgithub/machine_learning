import os
import shutil

import pandas as pd
import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt
from scipy import stats

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class MlBase(object):
    def __init__(self, target_variable):
        self.df = None
        self.df_continous = None
        self.df_categorical = None
        self.target_variable = target_variable
        if os.path.exists(
                os.path.join(os.path.dirname(__file__), os.pardir, 'plots', self.plt_folder)
        ):
            shutil.rmtree(
                os.path.join(os.path.dirname(__file__), os.pardir, 'plots', self.plt_folder)
            )

        if not os.path.exists(
                os.path.join(os.path.dirname(__file__), os.pardir, 'plots', self.plt_folder)
        ):
            os.makedirs(
                os.path.join(os.path.dirname(__file__), os.pardir, 'plots', self.plt_folder),
                exist_ok=True
            )

    def set_dafa_frame_from_csv(self, csv=None):
        """Sets data frame based on the csv passed

        :param string csv: 'path of csv file

        :returns pd.dataFrame: sets data frame based on the csv passed

        """
        self.df = pd.read_csv(csv)

    def get_value_counts(self, df, column):
        """Returns value counts

        :param string csv: 'path of csv file

        :returns pd.dataFram: Returns data frame based on the csv passed

        """
        return df[column].value_counts()

    def get_continuous_features(self, df):
        """Returns continuous features of dataframe

        :returns pd.dataFrame: Returns continuous features for the the data frame

        """
        # Extract continuous features in new dataframe
        df_copied = df.select_dtypes(exclude=['object']).copy()
        return df_copied.replace(' ', '_', regex=True)

    def get_categorical_features(self, df):
        """Returns categorical features of dataframe

        :returns pd.dataFrame: Returns categorical features for the the data frame

        """
        # Extract categorical features
        return df.select_dtypes(include=['object']).copy()

    def get_continuous_features_as_list(self, df):
        """Returns continuous features as list of dataframe

        :returns list: Returns continuous features for the the data frame as list

        """
        # Extract continuous features in new dataframe
        return df.select_dtypes(exclude=['object']).columns.tolist()

    def get_categorical_features_as_list(self, df):
        """Returns categorical features as list of dataframe

        :returns list: Returns categorical features for the the data frame as list

        """
        # Extract categorical features
        return df.select_dtypes(include=['object']).columns.tolist()

    def drop_columns(self, df, columns, inplace=False):
        """Drops columns from dataframe and returns the df

        :returns pd.dataFrame: Returns df with dropped df

        """
        return df.drop(columns, axis=1, inplace=inplace)

    def drop_empty_columns(self, df):
        """Drops columns from dataframe with all values as NAN (inplace)

        """
        # Get empty columns (with all values as Nan)
        empty_columns = df.columns[df.isnull().sum() == df.shape[0]].to_list()
        print('Dropping columns:', empty_columns)
        self.drop_columns(df, empty_columns, inplace=True)

    def drop_high_nan_columns(self, df, percent):
        """Drops columns with high values as NAN (inplace)

        """
        high_nan_columns = df.columns[
            df.isnull().sum() >= percent/100.0 * df.shape[0]
        ].to_list()
        print('Dropping columns with Nan > than %s : %s' % (percent, high_nan_columns))
        self.drop_columns(df, high_nan_columns, inplace=True)

    def remove_outliers(self, df):
        """Remove outliers (remove rows where zscore > 3 for any column)"""
        df_copied = df.copy()
        return df_copied[(np.abs(stats.zscore(df)) < 3).all(axis=1)]

    def get_n_valued_categorical_columns(self, df, n=1):
        categorical_cols = self.get_categorical_features_as_list(df)
        return df[categorical_cols].nunique()[df[categorical_cols].nunique() == n].index.tolist()

    def get_multi_valued_categorical_columns(self, df):
        categorical_cols = self.get_categorical_features_as_list(df)
        return df[categorical_cols].nunique()[df[categorical_cols].nunique() > 2].index.tolist()

    def transform_binary_columns(self, df):
        binary_colmmns = self.get_n_valued_categorical_columns(df, n=2)
        print('binary_colmmns', binary_colmmns)
        # Transform binary_columns in place
        for column in binary_colmmns:
            df[column] = LabelEncoder().fit_transform(df[column])

    def transform_multi_valued_columns(self, df, drop_first=True):
        multi_valued_colmmns = self.get_multi_valued_categorical_columns(df)
        print('multi_valued_colmmns', multi_valued_colmmns)
        dummies_adjusted_data = pd.get_dummies(
            data=df,
            columns=multi_valued_colmmns,
            drop_first=drop_first
        )
        return dummies_adjusted_data

    def get_scaled_df(self, df, type='min_max'):
        # Scaling Numerical columns
        if (type == 'min_max'):
            scaler = MinMaxScaler(feature_range=(0, 1))
        else:
            scaler = StandardScaler()

        numerical_cols = self.get_continuous_features_as_list(df)
        scaled_df = scaler.fit_transform(df[numerical_cols])
        # Important to use index same as df otherwise it starts indexing from 0
        scaled_df = pd.DataFrame(scaled_df, index=df.index.values, columns=numerical_cols)

        return scaled_df

    def get_df_shape(self, df):
        return df.shape

    def get_rows(self, df):
        return df.shape[0]

    def get_columns(self, df):
        return df.shape[1]

    def get_df_info(self, df):
        print(df.info())
        return df.info()

    def fill_missing_values_in_continuous(self, df, method='mean'):
        # Fill in values with average value for features missing values
        for col in df.columns:
            if (method == 'mean'):
                df[col].fillna(df[col].mean(), inplace=True)

    def fill_missing_values_in_categorical(self, df, level=0.1):
        num_rows = df.shape[0]
        cat_features = self.get_features_with_null_values(df)
        for column in cat_features:
            # If misising values > 10% of total values in column, Fill with value 'UNKNOWN'
            if column[1] > level * num_rows:
                df[column[0]].fillna('UNKNOWN', inplace=True)
            else:
                # If misising values <= 10% of total values in column, Fill with value with max frequency
                df[column[0]].fillna(df[column[0]].value_counts().index[0], inplace=True)

    def get_features_with_null_values(self, df):
        columns_with_null_values = []
        for column in df.columns:
            if df[column].isnull().sum():
                columns_with_null_values.append((column, df[column].isnull().sum()))

        print('Columns with null values: ', columns_with_null_values)
        print('Number of columns with null values: ', len(columns_with_null_values))
        return columns_with_null_values

    def get_categorical_feature_stats(self, df):
        """Prints stats for categorical features and plots greaphs and stores them in plots folder

        """
        columns_with_null_values = []
        file_stats = open(
            r'%s/%s.txt' % (
                os.path.join(os.path.dirname(__file__), os.pardir, 'plots', self.plt_folder),
                'stats'
            ), 'w'
        )
        for column in df.columns:
            if (df[column].dtypes == 'object'):
                column_count = df[column].value_counts()
                sns.set(style="darkgrid")
                sns.barplot(column_count.index, column_count.values, alpha=0.9)
                plt.title('Frequency Distribution of %s' % column)
                plt.ylabel('Number of Occurrences', fontsize=12)
                plt.xlabel(column, fontsize=12)
                # plt.show()
                plt.savefig(
                    '%s/%s.png' % (
                        os.path.join(os.path.dirname(__file__), os.pardir, 'plots', self.plt_folder),
                        column
                    )
                )
                plt.close()
                print('Stats for ', column)
                print('#' * 30)
                print(df[column].value_counts())

                file_stats.write('Stats for %s' % column)
                file_stats.write('#' * 30)
                file_stats.write(str(df[column].value_counts()))

                print('Null values for %s = %s' % (column, df[column].isnull().sum()))
                file_stats.write(
                    'Null values for %s = %s' % (column, str(df[column].isnull().sum()))
                )
                print(
                    'Number of different values for %s = %s' % (
                        column, df[column].value_counts().count()
                    )
                )
                file_stats.write(
                    'Number of different values for %s = %s' % (
                        column, str(df[column].value_counts().count()))
                )
                print('Value with highest frequency "%s"' % df[column].value_counts().index[0])
                file_stats.write(
                    'Value with highest frequency "%s"' % str(df[column].value_counts().index[0])
                )
                print('#' * 30)

                if df[column].isnull().sum():
                    columns_with_null_values.append((column, df[column].isnull().sum()))

        file_stats.close()
        print('Columns with null values: ', columns_with_null_values)
        print('Number of columns with null values: ', len(columns_with_null_values))

    def get_dummy_variables(self, df):
        for column in df.columns:
            if (df[column].dtypes != 'object'):
                raise Exception('data frame has data type: %s' % df[column].dtypes)

        return pd.get_dummies(df, prefix_sep='_', drop_first=True)

    def get_columns_with_correlation_greater_than_x(self, df, max_correlation=0.8):
        corr_matrix = df.corr()
        cols_to_melt = corr_matrix.columns
        corr_matrix = corr_matrix.reset_index()
        corr_matrix_melted = corr_matrix.melt(id_vars='index', value_vars=cols_to_melt)
        corr_matrix_melted = corr_matrix_melted[
            corr_matrix_melted['index'] != corr_matrix_melted['variable']
        ]
        corrlations_above_threshold = corr_matrix_melted[corr_matrix_melted.value.abs() > max_correlation]
        columns_to_drop = corrlations_above_threshold['index'].to_list()
        columns_to_drop = list(set(columns_to_drop))
        print('columns with correlation > than "%s": %s' % (max_correlation, columns_to_drop))
        return columns_to_drop

    def plot_heatmap_continuous(self, df):
        # Explore heatmap of selected continuous features
        # Extracting selected continuous features dataset from selected housing_data_with_dummies
        # retrieved after backward elimination
        cont_df = df[df.columns[df.columns.isin(self.get_continuous_features(df))]]
        cont_df_for_heatmap = cont_df.copy()
        Y = cont_df_for_heatmap[self.target_variable]  # Dependent Variable
        X = cont_df_for_heatmap.drop(
            [self.target_variable],
            axis=1
        )  # Independent variables (continous features)
        fig = plt.figure(figsize=(len(X.columns), len(X.columns)))
        cor = cont_df_for_heatmap.corr()
        sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
        # plt.show()
        plt.savefig(
            '%s/%s.png' % (
                os.path.join(os.path.dirname(__file__), os.pardir, 'plots', self.plt_folder),
                'continuous_heatmap'
            )
        )
        plt.close(fig)

    def plot_heatmap_all(self, df):
        df_heatmap = df.copy()
        Y = df_heatmap[self.target_variable]  # Dependent Variable
        X = df_heatmap.drop([self.target_variable], axis=1)  # Independent variables (continous features)
        fig = plt.figure(figsize=(len(X.columns), len(X.columns)))
        cor = df_heatmap.corr()
        sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
        # plt.show()
        plt.savefig(
            '%s/%s.png' %(
                os.path.join(os.path.dirname(__file__), os.pardir, 'plots', self.plt_folder),
                'All_features_heatmap'
            )
        )
        plt.close(fig)

    def split_data(self, df, test_size=0.2, random_state=42):
        df_copied = df.copy()
        Y = df_copied[self.target_variable]
        X_with_dummies = self.drop_columns(df_copied, columns=[self.target_variable])
        # Split selected (continous + categorical features) of housing data set into 80/20 training and test data
        # X_with_dummies and Y comes from code above
        X_train, X_test, Y_train, Y_test = train_test_split(
            X_with_dummies,
            Y,
            test_size=test_size,
            random_state=random_state
        )  # random_state controls the shuffling applied to the data before applying the split

        return X_train, X_test, Y_train, Y_test


if __name__ == '__main__':
    # orginal_regression_sbmitted()
    # new_regression_with_scaler_and_correlation()
    pass

