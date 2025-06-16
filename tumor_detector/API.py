from fastapi import FastAPI, File, UploadFile, APIRouter
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import cv2
import io
from PIL import Image
import uvicorn

app = FastAPI()
brain_tumor_detector_router = APIRouter()
model = tf.keras.models.load_model("tumor_detector/brain_tumor_model.h5")


def is_possible_mri(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(img)

        
        if img_array.ndim == 3 and img_array.shape[2] == 3:
            r, g, b = img_array[..., 0], img_array[..., 1], img_array[..., 2]
            if np.allclose(r, g, atol=15) and np.allclose(g, b, atol=15):
                
                w, h = img.size
                aspect_ratio = w / h
                if 0.8 < aspect_ratio < 1.2:
                    return True
        return False
    except:
        return False


def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = np.array(image)
    image = cv2.resize(image, (128, 128))
    image = image * (1. / 255)
    image = np.expand_dims(image, axis=0)
    return image


@brain_tumor_detector_router.post("/detect_tumor", response_class=JSONResponse)
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        
        if not is_possible_mri(contents):
            return JSONResponse({
                "success": False,
                "error": "The image is not suitable. Please make sure it is a brain scan"
            }, status_code=400)

        image = preprocess_image(contents)
        prediction = model.predict(image)
        probability = float(prediction[0][0])

        prediction_class = "TUMOR" if probability >= 0.5 else "NO TUMOR DETECTED"
        probability_percent = round(probability * 100, 2)

        if probability >= 0.85:
            comment = "There is a high chance of a tumor. Please seek further medical evaluation."
        elif probability >= 0.5:
            comment = "There is a moderate chance of a tumor. Recommend consulting a doctor."
        elif probability >= 0.15:
            comment = "There is a low possibility of a tumor. Medical confirmation is still recommended."
        else:
            comment = "Very low possibility of a tumor. No immediate concerns detected."

        return {
            "success": True,
            "prediction": prediction_class,
            "tumor probability ": f"{probability_percent}%",
            "Analysis": comment
        }

    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)





