import pickle
import numpy as np
from flask import Flask, request
from sklearn.linear_model import LinearRegression

with open('model.pkl', 'rb') as pkl_file:
    model = pickle.load(pkl_file)

app = Flask(__name__)


def model_prediction(value):
    return model.predict(np.array(value).reshape(-1,1))

@app.route('/predict')
def predict_func():
    value = request.args.get('value') 
    if type(int(value)) == int:
        prediction = model_prediction(int(value))
        return f'Результат: {prediction}'
    else:    
        return f'Нужно ввести число'
    
    
    
    
    
    
if __name__ == '__main__':
    app.run('localhost', 5000)
    