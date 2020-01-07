import numpy as np
from sklearn.linear_model import LogisticRegression
from exam import hours_studied_scaled, passed_exam, exam_features_scaled_train, exam_features_scaled_test, passed_exam_2_train, passed_exam_2_test, guessed_hours_scaled

# Create and fit logistic regression model here

model = LogisticRegression()
model.fit(hours_studied_scaled, passed_exam)
# Save the model coefficients and intercept here
calculated_coefficients = model.coef_
intercept = model.intercept_

print(calculated_coefficients, intercept)
# Predict the probabilities of passing for next semester's students here
passed_predictions = model.predict_proba(guessed_hours_scaled)

# Create a new model on the training data with two features here
model_2 = LogisticRegression()
model_2.fit(exam_features_scaled_train, passed_exam_2_train)
# Predict whether the students will pass here
passed_predictions_2 = model_2.predict(exam_features_scaled_test)

#Breast cancer model learing
import codecademylib3_seaborn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

cancer = load_breast_cancer()
X_train,X_test,Y_train,Y_test = train_test_split(cancer.data,cancer.target, train_size=.75, random_state=0)

model = LogisticRegression()
model.fit(X_train, Y_train)

predictions = model.predict(X_test)
score = model.score(X_test, Y_test)
#print(score)

#print(classification_report(Y_test, predictions))
print(X_test[1])
predict=model.predict([X_test[1]])
preds = cancer.target_names[predict]  # mapping the output label with the meaning of label.
print(preds)
