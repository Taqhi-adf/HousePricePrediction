df = pd.read_csv('Cleaned_house_price 20251213_182539.csv',index_col=0)
df.head()
df.isnull().sum()
df['House_type'].fillna(method='bfill',inplace=True)
df['Garage'].fillna(method='bfill',inplace=True)
df.isnull().sum()
df.info()

df['Year_built'] = df['Year_built'].astype('int64')
df['Year_built']

df.drop(columns=['City', 'House_type','Garage'],axis=1,inplace=True)
X = df[['Year_built','Area_in_Sqft','Bedrooms','Bathrooms','Listing_Date','House_Age']]
y = df['Price'] 


# Model for machine learning using regression problem
import numpy as np   # import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=42)

# model1
m1 = LinearRegression()
m1.fit(X_train,y_train)

y_pred1 = m1.predict(X_test)

mse1 = mean_squared_error(y_test, y_pred1)
mae1 = mean_absolute_error(y_test, y_pred1)
rmse1 = np.sqrt(mse1)
r21 = r2_score(y_test, y_pred1)  

print('Mean squared error: ',mse1)
print('Mean absolute error: ',mae1)
print('Root mean squared error:',rmse1)
print('R2 Score: ',r21)

# model2
from sklearn.tree import DecisionTreeRegressor, plot_tree
# Train Decision Tree Regressor
m2 = DecisionTreeRegressor(criterion='squared_error', max_depth=5)
m2.fit(X_train,y_train)

# Predictions
y_pred2 = m2.predict(X_test)

# Evaluation metrics
mse2 = mean_squared_error(y_test, y_pred2)
mae2 = mean_absolute_error(y_test, y_pred2)
rmse2 = np.sqrt(mse2)
r22 = r2_score(y_test, y_pred2)
print(f"Mean Squared Error: {mse2:.3f}")
print(f'Mean absolute error: {mae2:.3f}')
print(f"Root Mean Squared Error: {rmse2:.3f}")
print(f"R² Score: {r22:.3f}")

# model3
# Initialize Random Forest model
m3 = RandomForestRegressor(n_estimators=100, random_state=42)
m3.fit(X_train, y_train)

# Predict on test data
y_pred3 = m3.predict(X_test)

# Evaluate model
mse3 = mean_squared_error(y_test, y_pred3)
mae3 = mean_absolute_error(y_test, y_pred3)
rmse3 = np.sqrt(mse3)
r23 = r2_score(y_test, y_pred3)

print(f"Mean Squared Error: {mse3:.3f}")
print(f'Mean absolute error: {mae3:.3f}')
print(f"Root Mean Squared Error: {rmse3:.3f}")
print(f"R² Score: {r23:.3f}")

# model4
from xgboost import XGBRegressor
# Model
m4 = XGBRegressor(n_estimators=500,learning_rate=0.05,max_depth=6,random_state=42)
m4.fit(X_train, y_train)

# Predict on test data
y_pred4 = m4.predict(X_test)

# Evaluate model
mse4 = mean_squared_error(y_test, y_pred4)
mae4 = mean_absolute_error(y_test, y_pred4)
rmse4 = np.sqrt(mse4)
r24 = r2_score(y_test, y_pred4)

print(f"Mean Squared Error: {mse4:.3f}")
print(f'Mean absolute error: {mae4:.3f}')
print(f"Root Mean Squared Error: {rmse4:.3f}")
print(f"R² Score: {r24:.3f}")

# compare the models
m1_results = pd.DataFrame([['LinearRegression',mse1,mae1,rmse1,r21]],
                          columns= ['Method','Mean squared error','Mean absolute error','Root mean squared error','R2 Score'])
m2_results = pd.DataFrame(
    [['DecisionTreeRegressor', mse2, mae2, rmse2, r22]],
    columns=['Method', 'Mean squared error', 'Mean absolute error', 'Root mean squared error', 'R2 Score'])

m3_results = pd.DataFrame([['RandomForestRegressor',mse3,mae3,rmse3,r23]],
                          columns= ['Method','Mean squared error','Mean absolute error','Root mean squared error','R2 Score'])
m4_results = pd.DataFrame([['XGBoostRegressor',mse4,mae4,rmse4,r24]],
                          columns=['Method','Mean squared error','Mean absolute error','Root mean squared error','R2 Score'])

df_models = pd.concat([m1_results,m2_results,m3_results,m4_results],axis=0)
df_models

df_models.reset_index()
# the end