# Importing important libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import plotly.express as px

# Importing the dataset
data = pd.read_csv(r"C:/Users/varun/OneDrive/Documents/Power Pulse/individual+household+electric+power+consumption.zip", sep = ";")
data.head()

# Checking missing values if any
data[data['Global_active_power'] == '?']

# Checking the data types
data.info()

# Handling missing values
data.dropna(inplace = True)

data.columns

# Data set after cleaning missing values
data.isnull().sum()

# Checking duplicate values if any
data.duplicated().sum()

# Checking the shape of the data
data.shape

# Converting the artibute type 
data['Date'] = pd.to_datetime(data['Date'])
data['Date'] = data['Date'].dt.date
data['Global_active_power'] = pd.to_numeric(data['Global_active_power'])
data['Global_reactive_power'] = pd.to_numeric(data['Global_reactive_power'])
data['Voltage']=pd.to_numeric(data['Voltage'])
data['Global_intensity']=pd.to_numeric(data['Global_intensity'])
data['Sub_metering_1']=pd.to_numeric(data['Sub_metering_1'])
data['Sub_metering_2']=pd.to_numeric(data['Sub_metering_2'])

# Checking the information
data.info()

# Convert Date to datetime objects
data['Date'] = pd.to_datetime(data['Date'])

# Creating additional features
data['Datetime'] = pd.to_datetime(data['Date'].dt.strftime('%Y-%m-%d') + ' ' + data['Time'])

data['hour'] = data['Datetime'].dt.hour
data['day_of_week'] = data['Datetime'].dt.dayofweek
data['month'] = data['Datetime'].dt.month
data['year'] = data['Datetime'].dt.year
data['is_weekend'] = data['Datetime'].dt.dayofweek.isin([5,6]).astype(int)

data.drop(['Datetime'], axis = 1, inplace = True)

# Visualizing missing values
plt.figure(figsize=(10,6))
sns.heatmap(data.isnull(), yticklabels=False, cbar=True, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

numeric_columns = ['Global_active_power','Global_reactive_power','Voltage','Global_intensity',
                   'Sub_metering_1','Sub_metering_2','Sub_metering_3']

# Box Plot
plt.figure(figsize=(15,5))
data.boxplot(column=numeric_columns)
plt.xticks(rotation=45)
plt.title('Box Plot of Numerical Variables')
plt.show()

# Heatmap for correlation
numerical_cols = data.select_dtypes(include=['float64','int32','int64']).columns
corr_matrix = data[numerical_cols].corr()
plt.figure(figsize=(10,6))
sns.heatmap(corr_matrix,annot=True,cmap='coolwarm',center=0)
plt.title('Correlation Matrix')
plt.show()

# Visualizing sub-parameters
# Hourly pattern
plt.figure(figsize=(15,5))
hourly_avg = data.groupby('hour')['Global_active_power'].mean()
sns.lineplot(x=hourly_avg.index, y= hourly_avg.values)
plt.title('Average Power Consumption by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Average Global Active Power')
plt.tight_layout()
plt.show()

# Daily Pattern
plt.figure(figsize=(15,5))
daily_avg = data.groupby('day_of_week')['Global_active_power'].mean()
sns.barplot(x=daily_avg.index, y=daily_avg.values)
plt.title('AVerage Power Consumption by Day of Week')
plt.xlabel('Day of week (0=Monday)')
plt.ylabel('Average Global Active Power')
plt.show()

# Monthly Pattern
plt.figure(figsize=(15,5))
monthly_avg = data.groupby('month')['Global_active_power'].mean()
sns.barplot(x=monthly_avg.index, y=monthly_avg.values)
plt.title('Average Power Consumption by Month')
plt.xlabel('Month')
plt.ylabel('Average Global Active Power')
plt.show()

# Normalize dataset
scaler = StandardScaler()
data.iloc[:,2:] = scaler.fit_transform(data.iloc[:,2:])

# Train Test Split
from sklearn.model_selection import train_test_split
var = data.iloc[:, 2:].drop(['Global_active_power'], axis = 1)
target = data['Global_active_power']
X_train, X_test, y_train, y_test = train_test_split(var, target, test_size = 0.2, random_state = 42)

# Linear Regression
model1 = LinearRegression()
model1.fit(X_train,y_train)
y_pred = model1.predict(X_test)
print('Root Mean Squared Error:', root_mean_squared_error(y_test,y_pred))
print('Mean Absoulte Error:', mean_absolute_error(y_test,y_pred))
print('R2 Score:' ,r2_score(y_test,y_pred))

import pickle
with open('linear_regressor.pkl','wb') as file:
    pickle.dump(model1, file)

with open('linear_regressor.pkl', 'rb') as file:
    model = pickle.load(file)
y_pred = model.predict(X_test)
print('Root Mean Squared Error:', root_mean_squared_error(y_test,y_pred))
print('Mean Absolute Error:', mean_absolute_error(y_test,y_pred))
print('R2 Score:', r2_score(y_test,y_pred))

# Neural Network
model2 = MLPRegressor()
model2.fit(X_train,y_train)
y_pred = model2.predict(X_test)
print('Root Mean Squared Error:', root_mean_squared_error(y_test,y_pred))
print('Mean Absolute Error:', mean_absolute_error(y_test,y_pred))
print('R2 Score:', r2_score(y_test,y_pred))

# Gradient Boosting Regressor
model = GradientBoostingRegressor()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print('Root Mean Squared Error:', root_mean_squared_error(y_test,y_pred))
print('Mean Absolute Error',mean_absolute_error(y_test,y_pred))
print('R2 Score:', r2_score(y_test,y_pred))

# Random Forest Regressor
model = RandomForestRegressor()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print('Root Mean Squared Error:', root_mean_squared_error(y_test,y_pred))
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('R2 Score:', r2_score(y_test,y_pred))