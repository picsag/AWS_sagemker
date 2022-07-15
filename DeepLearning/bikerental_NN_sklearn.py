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

train_file = '../data/byke sharing/bike_train.csv'
validation_file = '../data/byke sharing/bike_validation.csv'
test_file = '../data/byke sharing/bike_test.csv'

# One Hot Encode all Categorical Features
# Let's define all the categorical features
categorical_features = ['season', 'holiday', 'workingday', 'weather', 'year', 'month', 'dayofweek', 'hour']

# Standardize Features
standardize_features = ['temp', 'atemp', 'humidity', 'windspeed']

columns = ['count',  'season',  'holiday',  'workingday',   'weather',  'temp',  'atemp',  'humidity',  'windspeed',
           'year',  'month', 'day',  'dayofweek',  'hour']

# Specify the column names as the file does not have column header
df_train = pd.read_csv(train_file,names=columns)
df_validation = pd.read_csv(validation_file,names=columns)

print(df_train.head())

print(df_validation.head())

X_train = df_train.iloc[:, 1:] # Features: 1st column onwards
y_train = df_train.iloc[:, 0].ravel() # Target: 0th column

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



# print(X_train_encoded.head())

