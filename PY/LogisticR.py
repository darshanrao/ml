
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return (1/(1+ np.exp(-x)))

x = np.linspace(-1,1,100)

sigmoid(x)

plt.plot(x , sigmoid(20 * x))

def sigmoid(x):
    return (1- np.exp(-x))/(1+ np.exp(-x))

plt.plot(x , sigmoid(20 * x))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
np.random.seed(0)

def gen_target(X):
    return X > 0.5

n_records = 300
X =np.sort(np.random.rand(n_records))
y =gen_target(X) 
X = X.reshape(-1,1)

# import pandas as pd
# data = pd.read_csv('Social_Network_Ads.csv')
# data.head()

# gen = {'Male': 1,'Female': 0}
# data.replace(gen, inplace = True)
# data.info()

# X = data[['EstimatedSalary']].values
# y = data['Purchased'].values

print('Number of training examples: ',X.shape[0])
print('Number of predictors: ', y.shape[1] if len(y.shape)>1 else 1)

X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.2)

lr = LogisticRegression()

lr.fit(X_train, y_train)
 
y_pred =lr.predict(X_test)

print("Coefficients: \n")

for ii, coef in enumerate(lr.coef_):
    for jj, c in enumerate(coef):
        print('Coeff-{0:2d}: {1:2.4f}'.format(jj, c))


print('\nAccuracy score: {:2.4f}'.format(accuracy_score(y_test, y_pred)))

if X.shape[1]< 2:
    plt.figure()
    plt.scatter(X_test, y_test )
    plt.scatter(X_test, y_pred,c = 'red')
    plt.show()