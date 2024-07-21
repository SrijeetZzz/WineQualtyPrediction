# Importing Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from warnings import filterwarnings
filterwarnings(action='ignore')

# Loading Dataset

wine = pd.read_csv("/content/winequality-red.csv")
print("Successfully Imported Data!")
print(wine)

sns.heatmap(wine.isnull())

## Distplot:

sns.distplot(wine['alcohol'])

## Histogram

wine.hist(figsize=(10,10),bins=50)
plt.show()

## Pair Plot:

sns.pairplot(wine,hue="quality")

X = wine.drop('quality',axis=1)
Y=wine['quality']

print(Y)

# Feature Importance

# Splitting Dataset

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3)

# LogisticRegression:

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix
print("Accuracy Score:",accuracy_score(Y_test,Y_pred))

from sklearn.metrics import classification_report
cr=classification_report(Y_test,Y_pred)
print(cr)

confusion_mat = confusion_matrix(Y_test,Y_pred)
print(confusion_mat)

# Using KNN:

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,y_pred))

# Using GaussianNB:

from sklearn.naive_bayes import GaussianNB
model3 = GaussianNB()
model3.fit(X_train,Y_train)
y_pred3 = model3.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,y_pred3))