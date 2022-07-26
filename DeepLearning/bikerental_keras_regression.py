import sys
import numpy as np


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Column Transformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, KBinsDiscretizer

# Keras Library
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping

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

print(df_validation.head())\

X_train = df_train.iloc[:,1:] # Features: 1st column onwards
y_train = df_train.iloc[:,0].ravel() # Target: 0th column

X_validation = df_validation.iloc[:,1:]
y_validation = df_validation.iloc[:,0].ravel()

colTransformer = ColumnTransformer([('onehot',
                                     OneHotEncoder(categories='auto',sparse=False),
                                     categorical_features),
                                    ('onehotday',
                                     OneHotEncoder(categories=[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]],
                                                   sparse=False),
                                     ['day']),
                                    ('standardize',
                                    StandardScaler(),standardize_features)
                                   ],
                                   remainder="passthrough")
colTransformer.fit(X_train)


X_train_encoded = colTransformer.transform(X_train)
X_validation_encoded = colTransformer.transform(X_validation)

print('Training Data',X_train.shape, 'OneHot Encoded',X_train_encoded.shape)
print('Val Data',X_validation.shape, 'OneHot Encoded',X_validation_encoded.shape)

# Dimension of input data
# We need to specify number of features when configuring the first hidden layer
print(X_train_encoded.shape)

model = Sequential()
# 1 hidden layer with 100 neurons with relu activation
# output layer - regression, so no activation
model.add(Dense(100, input_dim=X_train_encoded.shape[1],activation='relu'))
model.add(Dense(1, activation=None))

# Need to compile the model, specify the optimizer and loss function to use
# For a mean squared error regression problem
model.compile(optimizer='adam',
              loss='mse')

# We can optionally configure early stopping to prevent overfitting - stop when validation loss does not improve
early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)

history = model.fit(X_train_encoded, y_train, epochs=20, batch_size=32,
          validation_data=(X_validation_encoded, y_validation),callbacks=[early_stopping])

plt.scatter(x=history.epoch,y=history.history['loss'],label='Training Error')
plt.scatter(x=history.epoch,y=history.history['val_loss'],label='Validation Error')
plt.grid(True)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Vs Validation Error')
plt.legend()
plt.show()

# Compare actual vs predicted performance with dataset not seen by the model before
df = pd.read_csv(validation_file,names=columns)

print(df.head())

result = model.predict(X_validation_encoded)

df['count_predicted'] = result

print(df['count_predicted'].describe())

df['count'] = df['count'].map(np.expm1)
df['count_predicted'] = df['count_predicted'].map(np.expm1)

# Actual Vs Predicted
plt.plot(df['count'], label='Actual')
plt.plot(df['count_predicted'],label='Predicted')
plt.xlabel('Sample')
plt.ylabel('Count')
plt.xlim([100,150])
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

print("RMSE: {0:.2f}".format(mean_squared_error(df['count'],
                                                    df['count_predicted'])**.5))


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
result = model.predict(colTransformer.transform(X_test))

# Convert result to actual count
df_test["count"] = np.expm1(result)

print(df_test.head())

df_test[df_test["count"] < 0]

df_test[['datetime', 'count']].to_csv('predicted_count_keras.csv', index=False)
