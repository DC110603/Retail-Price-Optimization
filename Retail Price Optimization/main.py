#IMPORT THE NECESSARY PACKAGES

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

#LOAD THE DATASET

data=pd.read_csv('retail_data.csv')
print(data.head())

# Assuming the dataset has columns 'total_price', 'quantity', and 'actual_price/unit_price'
# You may need to adjust the column names based on your dataset

#1. Total Price Distribution Graph
plt.figure(figsize=(10, 6))
sns.histplot(data['total_price'], bins=30, kde=True)
plt.title('Total Price Distribution')
plt.xlabel('Total Price')
plt.ylabel('Count')
plt.show()

# 2. Relationship between Quantity of Products Sold and Total Price
plt.figure(figsize=(10, 6))
sns.scatterplot(x='quantity', y= 'total_price', data=data)
plt.title('Total Price vs Quantity Sold')
plt.xlabel('Quantity Sold')
plt.ylabel('Total Price')
plt.show()

# 3. Prepare data for Random Forest
# Assuming 'total_price' is the target variable and other columns are features
# Check if 'actual_price' column exists before dropping
if 'actual_price' in data.columns:
    X = data.drop(['total_price', 'actual_price'], axis=1) # Features
else:
    # Simply drop 'total_price' without printing the warning
    X = data.drop(['total_price'], axis=1)  # Features
y = data['total_price']  # Target variable
    # Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Create a LabelEncoder object
label_encoder=LabelEncoder()
for col in X_train.select_dtypes(include=['object']).columns:
    X_train[col] = label_encoder.fit_transform(X_train[col])
    X_test[col] = label_encoder.transform(X_test[col])  # Apply the same encoding to X_test
    # Iterate through all columns in X_train and encode object (string) columns
#4. Train the Random Forest model
model=RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train,y_train)

#5. Make predictions
y_pred=model.predict(X_test)

# 6. Retail Price vs Actual Price
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--') # Line for perfect prediction
plt.title('Retail Price vs Actual Price')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Retail Price')
plt.show()

# 7. Calculate accuracy percentage
mse=mean_squared_error(y_test, y_pred)
r2=r2_score(y_test, y_pred)
accuracy_percentage = r2 * 100
print (f'Mean Squared Error: {mse}')
print (f'R^2 Score: {r2}')
print(f'Accuracy Percentage: {accuracy_percentage:.2f}%')