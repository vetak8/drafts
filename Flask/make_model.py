from flask import Flask, request
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
import numpy as np
import pickle

X,y = load_diabetes(return_X_y=True)
