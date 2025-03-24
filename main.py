import numpy as np
import pickle
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = FastAPI()

app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
)

@app.post('/predict')
def predict_class(sepal_length, sepal_width, petal_length, petal_width):
        sepal_length = float(sepal_length)
        sepal_width = float(sepal_width)
        petal_length = float(petal_length)
        petal_width = float(petal_width)
        sample_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]]) # sepal_length, sepal_width, petal_length, petal_width
        predicted_class = model.predict(sample_data)

        return(f"Predicted class: {predicted_class[0]}")
