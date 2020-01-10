import codecademylib3_seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the passenger data
passengers = pd.read_csv('passengers.csv')

# Update sex column to numerical
passengers['Sex'] = passengers['Sex'].map({ 'male': 0, 'female': 0 })

# Fill the nan values in the age column

passengers['Age'].fillna(value=round(passengers.mean(axis = 0, skipna = True)['Age']),inplace=True)
#print(passengers['Age'].values)
# Create a first class column
passengers['FirstClass'] = passengers['Pclass'].apply(lambda x: 1 if x == 1 else 0)

# Create a second class column

passengers['SecondClass'] = passengers['Pclass'].apply(lambda x: 1 if x == 2 else 0)
#print(passengers)
# Select the desired features
selection =  passengers[['Sex','Age', 'FirstClass', 'SecondClass']]


survival = passengers['Survived']
# Perform train, test, split
#print(selection.shape, survival.shape)
X_train, X_test, y_train, y_test = train_test_split(selection, survival ,test_size = 0.8)

# Scale the feature data so it has mean = 0 and standard deviation = 1
scaler = StandardScaler()
train_features = scaler.fit_transform(X_train)
test_features = scaler.transform(X_test)

# Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)
# Score the model on the train data

model.score(X_train, y_train)
# Score the model on the test data

model.score(X_test, y_test)
# Analyze the coefficients

#print(model.coef_)
#print(list(zip(['Sex','Age','FirstClass','SecondClass'],model.coef_[0])))
# Sample passenger features
Jack = np.array([0.0,20.0,0.0,0.0])
Rose = np.array([1.0,17.0,1.0,0.0])
You = np.array([0.0,40.0,1.0,0.0])

# Combine passenger arrays
combined_arrays = np.array([ Jack , Rose, You ])

# Scale the sample passenger features
sample_passengers = scaler.transform(combined_arrays)

# Make survival predictions!
print(model.predict(sample_passengers))
print(model.predict_proba(sample_passengers))
