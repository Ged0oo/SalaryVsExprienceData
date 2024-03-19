import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

import os
os.chdir(r"E:\New folder\Python")
data = pd.read_csv("Salary_Data.csv")

data.info()
data.describe()

x = data.iloc[ : , :-1 ]
y = data.iloc[ : , 1 ]
xTrain, xTest, yTrain, yTest = train_test_split(x, y, train_size=0.8, random_state=10)

myModel = LinearRegression()
myModel.fit(xTrain, yTrain)
myModel.score(xTrain, yTrain)
myModel.score(xTest, yTest)

yPred = myModel.predict(xTrain)
plt.scatter(xTrain, yTrain, color='red')
plt.plot(xTrain, myModel.predict(xTrain), color='green')
plt.title('(Training Set) Salary Vs Experince')
plt.xlabel('Years of Experince')
plt.ylabel('Salary')
plt.show()

yPred = myModel.predict(xTest)
plt.scatter(xTest, yTest, color='red')
plt.plot(xTest, yPred, color='green')
plt.title('(Test Set) Salary Vs Experince')
plt.xlabel('Years of Experince')
plt.ylabel('Salary')
plt.show()

c = [i for i in range (1, len(yTest)+1)]
plt.plot(c, yTest, color='r', linestyle='-')
plt.plot(c, yPred, color='b', linestyle='-')
plt.title("Prediction")
plt.xlabel('Salary')
plt.ylabel('index')
plt.show()

mse = mean_squared_error(yTest, yPred)
rsq = r2_score(yTest, yPred)
print('Mean Squared Error : ', mse)
print("r Square : ", rsq)