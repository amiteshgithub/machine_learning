import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier

from algos.classifier_base import do_eda

from algos.classifier_base import ClassificationBase
from constants import TELCO_CHURN_DATA


class MlKerasClassification(ClassificationBase):
    def __init__(self, target_variable):
        self.model = None
        self.param_grid = {
            'tfidf__ngram_range': [(1,1), (1,2), (2,2), (1,3)],
            'tfidf__use_idf': [True, False],
            'kc__epochs': [10, 100, ],
            'kc__dense_nparams': [32, 256, 512],
            'kc__init': [ 'uniform', 'zeros', 'normal', ],
            'kc__batch_size':[2, 16, 32],
            'kc__optimizer':['RMSprop', 'Adam', 'Adamax', 'sgd'],
            'kc__dropout': [0.5, 0.4, 0.3, 0.2, 0.1, 0]
        }
        self.plt_folder = 'keras_classifier'
        super(MlKerasClassification, self).__init__(target_variable)

    # Function to create model, required for KerasClassifier
    def create_model(self, neurons=12, activation='relu', learn_rate=0.01, momentum=0):

        # create model
        model = Sequential()
        model.add(
            Dense(
                self.get_df_shape(self.df_with_dummies)[1],
                input_dim=self.get_df_shape(self.df_with_dummies)[1],
                activation=activation
            )
        )

        model.add(Dense(32, activation=activation))
        model.add(Dense(1, activation='sigmoid'))
        # Compile model
        optimizer = SGD(lr=learn_rate, momentum=momentum)
        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        return model

def keras_classifier(data_file, target_variable):
    # create object with target variable name
    ml = MlKerasClassification(target_variable)

    # Do EDA
    ml.df_with_dummies = do_eda(ml, data_file)

    # Here we create model after EDA because we need shape of df
    ml.model = KerasClassifier(
        build_fn=ml.create_model,
        epochs=50,
        batch_size=10,
        verbose=0
    )

    # Check cross validation scores
    # ml.check_cross_validation_scores(df_with_dummies, cv=5, scoring='r2')
    ml.check_cross_validation_scores(ml.df_with_dummies, cv=5)

    # predict on split data
    ml.plot_roc_curve_and_feature_importance(ml.df_with_dummies)

if __name__ == '__main__':
    keras_classifier(TELCO_CHURN_DATA, 'Churn') # with scaler and removal of high correlated features
