# New-Project.ML--II
This Machine Learning Project gives the prediction of using Social Media apps like WhatsApp, Facebook, Instagram etc.

Required Libraries 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

CREATING A SAMPLE DATASET 

data = {
    'Daily_Usage_Hours': [1, 2, 3, 4, 5, 6, 2, 7, 1, 8, 4, 5, 3, 6, 7],
    'Posts_Per_Week': [1, 2, 3, 5, 8, 10, 2, 12, 0, 15, 4, 9, 5, 8, 11],
    'Likes_Per_Day': [5, 15, 30, 40, 50, 70, 10, 90, 3, 100, 25, 60, 35, 80, 85],
    'Age': [15, 17, 18, 20, 22, 24, 25, 28, 30, 19, 23, 21, 16, 27, 29],
    # Target column: 1 = Will Continue using, 0 = Will Reduce/Stop
    'Future_Use': [1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1]
}

df = pd.DataFrame(data)
df.head()

SPLIT DATA INTO FEATURE AND TARGET

X = df[['Daily_Usage_Hours', 'Posts_Per_Week', 'Likes_Per_Day', 'Age']]
y = df['Future_Use']

TRAIN TEST SPLIT 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

LOGISTIC REGRESSION a simple and effective Model for CLASSIFICATION

model = LogisticRegression()
model.fit(X_train, y_train)

PREDICTION.

y_pred = model.predict(X_test)

EVALUATING 

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

TESTING WITH A NEW USER'S DATASETS

new_user = np.array([[4, 6, 40, 20]])  # 4 hours/day, 6 posts/week, 40 likes/day, 20 years old
prediction = model.predict(new_user)

if prediction[0] == 1:
    print("✅ The user is likely to CONTINUE using social media apps.")
else:
    print("❌ The user is likely to REDUCE or STOP using social media apps.")

    OUTPUTS 

    Accuracy: 0.85
✅ The user is likely to CONTINUE using social media apps.


