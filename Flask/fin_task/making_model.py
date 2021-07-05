from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
import numpy as np
import pickle

X,y = load_diabetes(return_X_y=True)
X = X[:, 0].reshape(-1,1)
model = LinearRegression(n_jobs=-1, normalize=True)
model.fit(X, y)

with open ('model.pkl', 'wb') as output:
    pickle.dump(model, output)