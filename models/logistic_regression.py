import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import accuracy_score, classification_report

# read the dataset
data= pd.read_csv("framingham.csv")

# handle the missing value 
data=data.dropna()

# feature and target selection
X=data.drop("TenYearCHD",axis=1)
y=data["TenYearCHD"]

# train test and split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# create a logistic regression model
model= LogisticRegression(max_iter=10000)

# train the model
model.fit(X_train,y_train)

# pridiction 
y_pred=model.predict(X_test)

# evaluate the model 
accuracy= accuracy_score(y_test,y_pred)

print("Accuracy:",accuracy)
print(classification_report(y_test,y_pred))

# save the model using pickle
with open("models/heart_disease.pkl","wb") as file:
    pickle.dump(model,file)

print("Model saved succesfully")
