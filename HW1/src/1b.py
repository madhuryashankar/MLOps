import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

file_path = "sampregdata.csv"  
df = pd.read_csv(file_path)
display(df.info())
display(df.describe())
display(df.head())
correlation = df.corr()['y'].drop('y').sort_values(ascending=False)
print("Feature correlations with target (y):")
display(correlation)
best_feature = correlation.abs().idxmax()
print(f"Best predictor: {best_feature}")
X = df[[best_feature]]
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")
model_filename = "linear_regression_model.pkl"
joblib.dump(model, model_filename)
print(f"Model saved as {model_filename}")
train_data_filename = "train_data.csv"
test_data_filename = "test_data.csv"
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)
train_data.to_csv(train_data_filename, index=False)
test_data.to_csv(test_data_filename, index=False)
print(f"Training data saved as {train_data_filename}")
print(f"Testing data saved as {test_data_filename}")

#Best predictor: x4
#Mean Squared Error (MSE): 76.85266230805573
#R² Score: 0.2685542927902339