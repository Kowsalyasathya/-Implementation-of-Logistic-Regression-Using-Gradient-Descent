# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
.Use the standard libraries in python for finding linear regression.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Predict the values of array.

5.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

6.Obtain the graph.

## Program:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Kowsalya M
RegisterNumber:  212222230069
```
```
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

df=np.loadtxt("/content/ex2data1.txt",delimiter=',')
X=df[:,[0,1]]
y=df[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.show()

def sigmoid(z):
  return 1/(1+np.exp(-z))

plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  J=-(np.dot(y,np.log(h)) + np.dot(1-y,np.log(1-h))) / X.shape[0]
  grad=np.dot(X.T,h-y)/X.shape[0]
  return J,grad

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h))) / X.shape[0]
  return J

def gradient(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  grad=np.dot(X.T,h-y) / X.shape[0]
  return grad

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta = np.array([0,0,0])
res = optimize.minimize(fun=cost,x0=theta,args=(X_train,y),
                        method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
  x_min,x_max=X[:,0].min() - 1,X[:,0].max()+1
  y_min,y_max=X[:,1].min() - 1,X[:,0].max()+1
  xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),
                    np.arange(y_min,y_max,0.1))

  X_plot = np.c_[xx.ravel(),yy.ravel()]
  X_plot = np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot = np.dot(X_plot,theta).reshape(xx.shape)

  plt.figure()
  plt.scatter(X[y==1][:,0],X[y==1][:,1],label='admitted')
  plt.scatter(X[y==0][:,0],X[y==0][:,1],label='NOT admitted')
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()

plotDecisionBoundary(res.x,X,y)

prob = sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
  X_train = np.hstack((np.ones((X.shape[0],1)),X))
  prob = sigmoid(np.dot(X_train,theta))
  return (prob>=0.5).astype(int)

np.mean(predict(res.x,X)==y)
```
## Output:
### Array value of X:
![image](https://github.com/Kowsalyasathya/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118671457/80ba5615-85dc-4357-9d3a-10ea83fe93db)
### Array value of Y:
![image](https://github.com/Kowsalyasathya/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118671457/662f4ae3-7675-4720-b2e7-c307600db314)
### Exam 1- Score Graph:
![image](https://github.com/Kowsalyasathya/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118671457/88d9b7ae-e660-43bf-b522-51858b21ba22)
### Sigmoid function graph:
![image](https://github.com/Kowsalyasathya/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118671457/459521e1-d8c5-4c5e-8006-535d91ab938c)
### X_train_grad value:
![image](https://github.com/Kowsalyasathya/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118671457/da01e33e-726f-4b92-bd29-33fc1caf072a)
###  Y__train_grad value:
![image](https://github.com/Kowsalyasathya/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118671457/efdd19e8-6ad9-4c3d-855d-9eadd77e96c9)
### Print res.x:
![image](https://github.com/Kowsalyasathya/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118671457/d11c4f1a-1104-4f6e-9ebc-f16c965a6617)
### Decision Boundary-Graph for Exam Score:
![image](https://github.com/Kowsalyasathya/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118671457/46eaee2f-696e-4720-a1ee-99e84702b28f)
### Probability value:
![image](https://github.com/Kowsalyasathya/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118671457/13e52b70-9b10-4e1e-831d-d1cb6914c12e)
### Prediction value of mean:
![image](https://github.com/Kowsalyasathya/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118671457/15ae63f3-d44b-4be0-b2ff-b6320092b31b)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

