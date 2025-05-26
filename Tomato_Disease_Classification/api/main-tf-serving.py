import uvicorn
from cffi.cffi_opcode import CLASS_NAME
from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests
# from starlette.middleware.cors import CORSMiddleware

app = FastAPI()
# origins = [
#     "http://localhost",
#     "http://localhost:3000",
# ]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )



endpoint = "http://localhost:8501/v1/models/tomatoes_model:predict"

CLASS_NAMES = ['Tomato_Bacterial_spot',
 'Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_Leaf_Mold',
 'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite',
 'Tomato__Target_Spot',
 'Tomato__Tomato_YellowLeaf__Curl_Virus',
 'Tomato__Tomato_mosaic_virus',
 'Tomato_healthy']
@app.get("/ping")
async def ping():
    return "helloo"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
file:UploadFile = File(),
):
    image = read_file_as_image( await file.read())
    image_batch = np.expand_dims(image, axis=0)
    json_data = {
        "instances": image_batch.tolist()
    }
    response = requests.post(endpoint, json=json_data)
    prediction = response.json()['predictions'][0]
    predicted_class =  np.argmax(prediction)
    confidence = np.max(prediction)
    return {
        "class": CLASS_NAMES[predicted_class],
        "confidence": confidence
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)