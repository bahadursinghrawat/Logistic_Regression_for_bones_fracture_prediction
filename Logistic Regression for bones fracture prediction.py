#!/usr/bin/env python
# coding: utf-8

# <h1>Logistic Regression for bones fracture prediction</h1>

# <h2>Problem statement</h2>

# We consider the dataset “Fracture.xlsx” containing information on some patients and their bone
# fracture states; fractured and unfractured.
# Some information on each patient (indicated by his id) is provided: age, sex, weight, height, and 
# bmd (bone mineral density). 
# We want to find the best logistic regression model that fits the dataset and predict the bone state
# based on the following features: age, sex, bmd, and bmi (body mass index). The value of the bmi
# feature can be computed as follows: 
# 
#     
#     bmi = weight/height^2
#     
#     
# Where weight must be in kg and height in meter

# <h2>To predict:</h2>

# In[42]:


#1.Use data visualization techniques to explain the relationship between features. 



#2.Find the logistic regression model that fits the data and predicts the bone fracture.


#3.Use cross-validation while building the model.


#4.Apply the selected model to test data and compute the accuracy, precision, recall, and F1 score.


# In[43]:




#Read the data from the provided Excel sheet


# In[44]:


import pandas as pd

# Read the Excel file
data = pd.read_excel('D:\\New folder\\Fracture.xlsx')

# Display the first few rows of the data
print(data.head(10))


# In[45]:


#Use data visualization techniques to explain the relationship between features


# In[46]:


import seaborn as sns
import matplotlib.pyplot as plt

# Create a pairplot to visualize the relationships between features
sns.pairplot(data, hue='fracture')
plt.show()


# In[47]:


#Find the logistic regression model that fits the data


# In[48]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Calculate BMI
data['bmi'] = data['weight_kg'] / ((data['height_cm']**2)*100)   # weight in kg, height in m


# Perform one-hot encoding on 'sex' column
data_encoded = pd.get_dummies(data, columns=['sex'])

# Prepare the features and target variable
X = data_encoded[['age', 'bmd', 'bmi', 'sex_F', 'sex_M']]
y = data_encoded['fracture']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of Logistic Regression model
model = LogisticRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)


# In[49]:


#Use cross-validation while building the model


# In[50]:


from sklearn.model_selection import cross_val_score

# Perform cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)

# Print the cross-validation scores
print("Cross-Validation Scores:", cv_scores)
print("Average Cross-Validation Score:", cv_scores.mean())


# In[51]:


#Compute the accuracy, precision, recall, and F1 score


# In[52]:


# Calculate accuracy, precision, recall, and F1 score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='fracture')
recall = recall_score(y_test, y_pred, pos_label='fracture')
f1 = f1_score(y_test, y_pred, pos_label='fracture')

# Print the accuracy, precision, recall, and F1 score
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




