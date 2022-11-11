
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split  # For evaluation
from sklearn.metrics import accuracy_score
np.random.seed(0)

from sklearn import datasets
X,y = datasets.load_iris(return_X_y=True)
# X = X[:,3].reshape(-1,1)

# def gen_target(X):
#     return (X) > 0.5
# n_records = 300         # Total number of records
# X = np.sort(np.random.rand(n_records))    # Randomly generate data points (features)
# y = gen_target(X)    # Generate regression output with additive noise
# X = X.reshape(-1,1)
# print('Number of training examples : ',X.shape[0])
# print('Number of predictors : ',y.shape[1] if len(y.shape)>1 else 1)

"""# Decision Tree Classifier"""

X_train ,X_test , y_train,y_test = train_test_split(X,y,test_size=0.2)
dt = DecisionTreeClassifier(max_depth=5,criterion='entropy')
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)
print('\nAccuracy Score: {:2.4f}'.format(accuracy_score(y_test,y_pred)))

if X.shape[1]<2:
  plt.figure()
  plt.scatter(X_train,y_train,color='red')
  plt.scatter(X_test,y_pred,color='blue',linewidth=3)
  plt.show()

dt.get_depth()

from sklearn import tree 
plt.figure(figsize=(5,5),dpi=150)
_=tree.plot_tree(dt,filled=True)

"""#Bagging"""

from sklearn.ensemble import BaggingClassifier 
X_train ,X_test , y_train,y_test = train_test_split(X,y,test_size=0.2)
bc = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=1)
bc.fit(X_train,y_train)
y_pred = bc.predict(X_test)
print('\nAccuracy Score: {:2.4f}'.format(accuracy_score(y_test,y_pred)))