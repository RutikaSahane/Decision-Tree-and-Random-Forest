# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#import dataset
dataset = pd.read_csv(r"E:\fsds_course\15. Logistic regression with future prediction\Social_Network_Ads.csv") 

x= dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values


#Split dataset in training and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.20,random_state=0)



# Feature Scaling
'''
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train=sc.fit_transform(x_train)
x_test =sc.transform(x_test)

'''
from sklearn.tree import DecisionTreeClassifier
classifier =DecisionTreeClassifier()
classifier.fit(x_train,y_train)

# predict the test set results
y_pred=classifier.predict(x_test)

#making confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
print(cm)


#Accuracy of model
from sklearn.metrics import accuracy_score
ac =accuracy_score(y_test,y_pred)
print(ac)


# this is to get the classification Report

from sklearn.metrics import classification_report
cr=classification_report(y_test, y_pred)
cr

bias = classifier.score(x_train,y_train)
bias

variance=classifier.score(x_test,y_test)
variance

