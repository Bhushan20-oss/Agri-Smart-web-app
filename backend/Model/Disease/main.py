from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np
import tensorflow as tf

app = FastAPI()

# Allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import io
import tensorflow as tf

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
try:
    model = tf.keras.models.load_model("trainedmodel.h5")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Class names
class_names = [
    "BACTERIAL LEAF BLIGHT", "BACTERIAL PANICAL BLIGHT", "BROWN SPOT", 
    "DEAD HEART", "HEALTHY", "LEAF BLAST", "LEAF SCALD", 
    "NARROW BROWN SPOT", "NECK BLAST", "RICE HISPA", 
    "SHEATH BLIGHT", "TUNGRO"
]

# Mapping class name to solution
solutions = {
    "BACTERIAL LEAF BLIGHT": "Use copper-based bactericides and resistant varieties.",
    "BACTERIAL PANICAL BLIGHT": "Improve drainage and avoid excessive nitrogen.",
    "BROWN SPOT": "Apply fungicides and use balanced fertilization.",
    "DEAD HEART": "Control early with appropriate insecticides.",
    "HEALTHY": "No action needed. Crop is healthy.",
    "LEAF BLAST": "Apply tricyclazole and avoid high nitrogen doses.",
    "LEAF SCALD": "Ensure proper field drainage and fungicide use.",
    "NARROW BROWN SPOT": "Use resistant varieties and seed treatments.",
    "NECK BLAST": "Apply fungicide before flowering stage.",
    "RICE HISPA": "Manual removal and spray insecticides.",
    "SHEATH BLIGHT": "Use validamycin and avoid dense planting.",
    "TUNGRO": "Remove infected plants and control green leafhopper."
}

# Image preprocessing
def read_imagefile(file) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(file)).convert("RGB")
        image = image.resize((224, 224))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        print(f"Received file: {file.filename}, Content-Type: {file.content_type}")

        if not file.content_type.startswith("image/"):
            return JSONResponse(status_code=400, content={"error": "File is not an image."})

        image_bytes = await file.read()
        img_array = read_imagefile(image_bytes)

        if img_array is None:
            return JSONResponse(status_code=400, content={"error": "Failed to process image."})

        print(f"Image shape: {img_array.shape}")

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        confidence = float(np.max(prediction)) * 100
        disease_name = class_names[predicted_class]
        solution = solutions.get(disease_name, "No solution available.")

        return JSONResponse(content={
            "disease": disease_name,
            "confidence": round(confidence, 2),
            "solution": solution
        })

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Prediction failed: {str(e)}"})

# Load your model
model = tf.keras.models.load_model("model.h5")

# Disease to solution mapping
disease_solutions = {
    "Bacterial Leaf Blight": "Use certified seeds, avoid water logging, and apply recommended bactericides.",
    "Brown Spot": "Apply fungicides like Mancozeb and ensure proper drainage.",
    "Leaf Smut": "Remove infected leaves and apply appropriate fungicide.",
    "Healthy": "Your crop appears healthy. Continue regular monitoring and maintenance."
}

# Prediction function
def read_imagefile(file) -> Image.Image:
    image = Image.open(io.BytesIO(file)).convert("RGB")
    image = image.resize((224, 224))  # Ensure same as model input
    return np.expand_dims(np.array(image) / 255.0, axis=0)

def model_predict(img_array):
    prediction = model.predict(img_array)
    class_names = ["Bacterial Leaf Blight", "Brown Spot", "Leaf Smut", "Healthy"]
    predicted_index = np.argmax(prediction)
    return class_names[predicted_index]

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = read_imagefile(await file.read())
    prediction = model_predict(image)
    solution = disease_solutions.get(prediction, "No solution found.")
    return {"disease": prediction, "solution": solution}
