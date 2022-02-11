import os

# CSV data files
EMPLOYEE_CHURN_DATA = (
    'https://raw.githubusercontent.com/cjflanagan/cs68/master/WA_Fn-UseC_-HR-Employee-Attrition.csv'
)
# HOUSING_DATA=('https://raw.githubusercontent.com/cjflanagan/cs68/master/housing.csv')

HOUSING_DATA=('%s' % (os.path.join(os.path.dirname(__file__), 'housing_data_train.csv')))

TELCO_CHURN_DATA = (
    '%s' % (os.path.join(os.path.dirname(__file__), 'Telco-Customer-Churn.csv'))
)

TELCO_CHURN_DATA_TEST = (
    '%s' % (os.path.join(os.path.dirname(__file__), 'Telco-Customer-Churn-Test.csv'))
)