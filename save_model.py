import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier




# Reading the csv
df = pd.read_csv('https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Data/classification_sprint/iris.csv')

# Separate into features and target
y = df['species']
X = df.drop('species', axis=1)

# Standardise the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_train_scaled, y, test_size=0.30, random_state=50)

# Train the model
forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(X_train, y_train)


# Save the model
with open('iris_model.pkl', 'wb') as model_file:
    pickle.dump(forest, model_file)
    
# Save the model
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)
    
print('Model and scaler saved successfully')