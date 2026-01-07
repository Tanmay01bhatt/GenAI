import uvicorn
from fastapi import FastAPI
from val import TextVal
import numpy as np
import pickle
import pandas as pd

model = pickle.load(open('trained_model.pkl', 'rb'))
tf_idf = pickle.load(open('tf_idf.pkl', 'rb'))

api = FastAPI(title="Cuisine Detector API",
    description="Predict cuisine from text input",)

@api.get('/')
def index():
    return {"message": "Cuisine Detector API is running"}

@api.post('/predict')
def predict_cuisine(data:TextVal):
    data = data.model_dump() 
    inp = data['text']
    inp = np.array([inp])
    inp = tf_idf.transform(inp)
    pred = model.predict(inp)

    return {"predicted_cuisine": pred}

if __name__ == '__main__':
    uvicorn.run(api,host='127.0.0.1',port=8000)