#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
num_records = 47377

# Generate synthetic data
crime_types = ['Theft', 'Assault', 'Robbery', 'Burglary', 'Vandalism', 'Drug Offense', 'Fraud']
locations = ['North', 'South', 'East', 'West', 'Downtown', 'Beach', 'Suburbs']

data = {
    'Crime_Type': np.random.choice(crime_types, num_records),
    'Time_of_Day': np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'], num_records),
    'Location': np.random.choice(locations, num_records),
    'Date': pd.date_range(start='2022-01-01', periods=num_records, freq='H'),
    'Reported_Crime': np.random.randint(1, 100, num_records)
}

# Create DataFrame
crime_data = pd.DataFrame(data)
crime_data['Hour'] = crime_data['Date'].dt.hour

# Save the dataset for further use
crime_data.to_csv('miami_crime_data.csv', index=False)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
crime_data = pd.read_csv('miami_crime_data.csv')

# Display summary statistics
print(crime_data.describe())

# Count plot for crime types
plt.figure(figsize=(10, 6))
sns.countplot(data=crime_data, x='Crime_Type', order=crime_data['Crime_Type'].value_counts().index)
plt.title('Crime Types Distribution')
plt.xticks(rotation=45)
plt.show()

# Heatmap for crimes by hour and location
crime_pivot = crime_data.pivot_table(index='Hour', columns='Location', values='Reported_Crime', aggfunc='sum', fill_value=0)
plt.figure(figsize=(12, 8))
sns.heatmap(crime_pivot, cmap='YlGnBu')
plt.title('Heatmap of Reported Crimes by Hour and Location')
plt.xlabel('Location')
plt.ylabel('Hour of Day')
plt.show()

# Time series analysis of crime
crime_data['Date'] = pd.to_datetime(crime_data['Date'])
crime_time_series = crime_data.groupby(crime_data['Date'].dt.date).sum()['Reported_Crime']

plt.figure(figsize=(12, 6))
crime_time_series.plot()
plt.title('Time Series of Reported Crimes in Miami')
plt.xlabel('Date')
plt.ylabel('Number of Reported Crimes')
plt.xticks(rotation=45)
plt.show()


# In[ ]:


# Check variable correlation
correlation = crime_data.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


# In[ ]:


# Encode categorical variables
crime_data_encoded = pd.get_dummies(crime_data, columns=['Crime_Type', 'Time_of_Day', 'Location'])


# In[ ]:


# Build decision tree classifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Features and target variable
X = crime_data_encoded.drop(columns=['Reported_Crime', 'Date', 'Hour'])
y = (crime_data['Reported_Crime'] > crime_data['Reported_Crime'].median()).astype(int)  # Binary classification

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Predictions
y_pred = dt_classifier.predict(X_test)

# Evaluation
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

