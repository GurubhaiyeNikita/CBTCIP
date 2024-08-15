import sklearn
import pandas as p
import numpy as n
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split

d=p.read_excel("Iris Flower.xlsx")

print("jsk")
d.describe

print(d.isna().sum())

d.head(10)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
d.shape



sepal_legth=d['SepalLengthCm'].values.tolist()
sepal_width=d['SepalWidthCm'].values.tolist()
petal_legth=d['PetalLengthCm'].values.tolist()
petal_width=d['PetalWidthCm'].values.tolist()
print("jsk")

X=d.drop(['Id','Species'], axis=1)
d['Class Species']=d['Species'].apply(lambda x:1 if x=='Iris-setosa' else 0 if x=='Iris-versicolor' else 2)


y=d['Species']
#y=p.factorize(d['Species'])[0].reshape[1,1]
print("jsk")
#x_train, x_test, y_train y_test = train_test_split(X,Y,test_size=0.25,random_state=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)


#print(y_train)
print("jsk")

c=sklearn.tree.DecisionTreeClassifier()

cc=c.fit(X_train,y_train)
print(cc)

t=[[1,2.1,3.1,1.2]]
#t=list(input("enter"))
#q=t.reshape(-1,1)
#z=t.tolist()


k=cc.predict(t)
print(k)


