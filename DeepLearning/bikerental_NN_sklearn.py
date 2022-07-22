import sys
import numpy as np
# Set random seed
np.random.seed(0)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# NN
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, KBinsDiscretizer
# Column Transformer
from sklearn.compose import ColumnTransformer

train_file = '../data/byke sharing/bike_train_log.csv'
validation_file = '../data/byke sharing/bike_validation_log.csv'
test_file = '../data/byke sharing/bike_test.csv'

# One Hot Encode all Categorical Features
# Let's define all the categorical features
categorical_features = ['season', 'holiday', 'workingday', 'weather', 'year', 'month', 'dayofweek', 'hour']

# Standardize Features
standardize_features = ['temp', 'atemp', 'humidity', 'windspeed']

columns = ['count',  'season',  'holiday',  'workingday',   'weather',  'temp',  'atemp',  'humidity',  'windspeed',
           'year',  'month', 'day',  'dayofweek',  'hour']

# Specify the column names as the file does not have column header
df_train = pd.read_csv(train_file, names=columns)
df_validation = pd.read_csv(validation_file,names=columns)

print(df_train.head())

print(df_validation.head())

X_train = df_train.iloc[:, 1:]  # Features: 1st column onwards
y_train = df_train.iloc[:, 0].ravel()  # Target: 0th column

X_validation = df_validation.iloc[:, 1:]
y_validation = df_validation.iloc[:, 0].ravel()

print(X_train.head())

colTransformer = ColumnTransformer([('onehot',
                                     OneHotEncoder(categories='auto', sparse=False),
                                     categorical_features),
                                    ('onehotday',
                                     OneHotEncoder(categories=[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]],
                                                   sparse=False),
                                     ['day']),
                                    ('standardize',
                                    StandardScaler(), standardize_features)
                                   ],
                                   remainder="passthrough")

colTransformer.fit(X_train)

y = pd.get_dummies(X_train, columns=categorical_features, drop_first=False)

print(y.head())

X_train_encoded = colTransformer.transform(X_train)
X_validation_encoded = colTransformer.transform(X_validation)

print('Training Data', X_train.shape, 'OneHot Encoded', X_train_encoded.shape)
print('Val Data', X_validation.shape, 'OneHot Encoded', X_validation_encoded.shape)

nn_regressor = MLPRegressor(random_state=5,
                            hidden_layer_sizes=[100],
                            activation='relu',
                            max_iter=100)


nn_regressor.fit(X_train_encoded, y_train)

# predict the count for validation data
df = pd.read_csv(validation_file, names=columns)
print(df.head())
result = nn_regressor.predict(X_validation_encoded)
df['count_predicted'] = result
print(df.head())
print(df['count_predicted'].describe())

# Convert log(count) to count
df['count'] = df['count'].map(np.expm1)
df['count_predicted'] = df['count_predicted'].map(np.expm1)

# Actual Vs Predicted
plt.plot(df['count'], label='Actual')
plt.plot(df['count_predicted'], label='Predicted')
plt.xlabel('Sample')
plt.ylabel('Count')
plt.xlim([100, 150])
plt.title('Validation Dataset - Predicted Vs. Actual')
plt.legend()
plt.show()

# Over prediction and Under Prediction needs to be balanced
# Training Data Residuals
residuals = (df['count'] - df['count_predicted'])

plt.hist(residuals)
plt.grid(True)
plt.xlabel('Actual - Predicted')
plt.ylabel('Count')
plt.title('Residuals Distribution')
plt.axvline(color='r')
plt.show()

value_counts = (residuals > 0).value_counts(sort=False)
print(' Under Estimation: {0:.2f}'.format(value_counts[True]/len(residuals)))
print(' Over  Estimation: {0:.2f}'.format(value_counts[False]/len(residuals)))

print("RMSE: {0:.2f}".format(mean_squared_error(df['count'], df['count_predicted'])**.5))


# Metric Use By Kaggle
def compute_rmsle(y_true, y_pred):
    if type(y_true) != np.ndarray:
        y_true = np.array(y_true)

    if type(y_pred) != np.ndarray:
        y_pred = np.array(y_pred)

    return np.average((np.log1p(y_pred) - np.log1p(y_true)) ** 2) ** .5


print("RMSLE: {0:.2f}".format(compute_rmsle(df['count'], df['count_predicted'])))

# Optional Test Data
# Prepare Data for Submission to Kaggle
df_test = pd.read_csv(test_file, parse_dates=['datetime'])

X_test = df_test.iloc[:, 1:]  # Exclude datetime for prediction

X_test.head()

# Transform data first with column transformer
result = nn_regressor.predict(colTransformer.transform(X_test))

# Convert result to actual count
df_test["count"] = np.expm1(result)

print(df_test.head())

df_test[df_test["count"] < 0]

df_test[['datetime', 'count']].to_csv('predicted_count.csv', index=False)



