import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor

# loading the data set 
data = pd.read_csv("NetflixOriginals.csv",encoding='latin-1')


# Development Part 1
print(data.columns)
print(data.head())
# getting the information about the data for bettter understanding
print(data.info())
# checking for any duplicate data in the data set
print(data.duplicated().any())

# encoding the categorical features
label_encoder = LabelEncoder()
data['Genre'] = label_encoder.fit_transform(data['Genre'])
data['Language'] = label_encoder.fit_transform(data['Language'])
# encoding the premiere data since they contain month date, year and month date. year
data['Premiere_comma'] = pd.to_datetime(data['Premiere'], format='%B %d, %Y', errors='coerce')
data['Premiere_period'] = pd.to_datetime(data['Premiere'], format='%B %d. %Y', errors='coerce')

# Merge the two columns into a single datetime column
data['Premiere'] = data['Premiere_comma'].combine_first(data['Premiere_period'])

# Drop the temporary columns
data.drop(['Premiere_comma', 'Premiere_period'], axis=1, inplace=True)

# Convert valid datetime values to timestamps and handle NaT values
data['Premiere'] = data['Premiere'].apply(lambda x: x.timestamp() if not pd.isna(x) else None)

data.head()

# checking for any missing data in the premiere
print(data.isnull().values.any())



# Development Part 2
# Split data into features (X) and target (y)
X = data[['Genre','Premiere','Runtime', 'Language']]
y = data['IMDB Score']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose a regression algorithm
model = LinearRegression()
# Initialize the Gradient Boosting Regressor
gb_regressor = GradientBoostingRegressor()
# Model Training
model.fit(X_train, y_train)
# Fit the model to your training data
gb_regressor.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
# Make predictions
predictions = gb_regressor.predict(X_test)

# by using LinearRegression method
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("by using LinearRegression method")
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# by using Gradient Boosting Regressor
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("by using Gradient Boosting Regressor")
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'R-squared: {r2}')    
    