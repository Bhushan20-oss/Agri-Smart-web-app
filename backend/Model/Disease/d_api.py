from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = FastAPI()

# CORS setup (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your trained model
model = tf.keras.models.load_model("trainedmodel.keras")

# Match class order with training
class_names = [
    "BACTERIAL LEAF BLIGHT",
    "BACTERIAL PANICAL BLIGHT",
    "BROWN SPOT",
    "DEAD HEART",
    "HEALTHY",
    "LEAF BLAST",
    "LEAF SCALD",
    "NARROW BROWN SPOT",
    "NECK BLAST",
    "RICE HISPA",
    "SHEATH BLIGHT",
    "TUNGRO"
]

descriptions = {
    "BACTERIAL LEAF BLIGHT": {
        "description": "Bacterial Leaf Blight causes yellowing and wilting of rice leaves, primarily due to Xanthomonas oryzae pv. oryzae.",
        "treatment": ["Use resistant varieties.", "Apply balanced fertilizers.", "Avoid over-irrigation."]
    },
    "BACTERIAL PANICAL BLIGHT": {
        "description": "A disease affecting rice panicles, causing discoloration and sterility.",
        "treatment": ["Plant resistant varieties.", "Manage nitrogen levels.", "Control vectors."]
    },
    "BROWN SPOT": {
        "description": "Caused by Bipolaris oryzae, resulting in brown lesions on leaves and poor grain quality.",
        "treatment": ["Use fungicides like Mancozeb.", "Ensure proper field drainage.", "Use certified seeds."]
    },
    "DEAD HEART": {
        "description": "Central shoot of young rice plants dries and dies due to stem borer infestation.",
        "treatment": ["Use pheromone traps.", "Apply insecticides like Chlorantraniliprole."]
    },
    "HEALTHY": {
        "description": "The plant is healthy with no visible disease symptoms.",
        "treatment": ["No action needed. Maintain good agricultural practices."]
    },
    "LEAF BLAST": {
        "description": "Caused by Pyricularia oryzae, leading to spindle-shaped lesions and leaf wilting.",
        "treatment": ["Use resistant varieties.", "Avoid excess nitrogen.", "Apply fungicides early."]
    },
    "LEAF SCALD": {
        "description": "Fungal disease causing brown to gray lesions, often starting from leaf tip.",
        "treatment": ["Improve air circulation.", "Apply appropriate fungicides."]
    },
    "NARROW BROWN SPOT": {
        "description": "Leptosphaeria salvinii causes narrow brown lesions on leaves and panicles.",
        "treatment": ["Use clean seed.", "Apply fungicides if severe."]
    },
    "NECK BLAST": {
        "description": "A severe form of blast affecting panicle necks, causing sterility.",
        "treatment": ["Use resistant varieties.", "Apply fungicides at booting stage."]
    },
    "RICE HISPA": {
        "description": "Insect pest that scrapes leaf tissues and damages photosynthesis.",
        "treatment": ["Spray with insecticides like Malathion.", "Avoid overcrowding."]
    },
    "SHEATH BLIGHT": {
        "description": "Caused by Rhizoctonia solani, it forms lesions on leaf sheaths and spreads rapidly.",
        "treatment": ["Ensure good drainage.", "Apply fungicides like Validamycin."]
    },
    "TUNGRO": {
        "description": "Viral disease spread by green leafhoppers causing stunted growth and yellow-orange discoloration.",
        "treatment": ["Control vectors.", "Use resistant varieties.", "Avoid overlapping crops."]
    }
}

# Image preprocessing function (no normalization)
def read_imagefile(file_bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        image = image.resize((128, 128))  # Match model input size
        image = np.array(image)  # No /255.0
        image = np.expand_dims(image, axis=0).astype(np.float32)
        return image
    except Exception as e:
        print(f"❌ Error reading image: {e}")
        return None

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        print(f"📦 Received file: {file.filename}, Content-Type: {file.content_type}")

        if not file.content_type.startswith("image/"):
            return JSONResponse(status_code=400, content={"error": "Invalid image file"})

        image_bytes = await file.read()
        image = read_imagefile(image_bytes)

        if image is None:
            return JSONResponse(status_code=400, content={"error": "Image preprocessing failed"})

        print(f"🖼️ Image preprocessed: shape={image.shape}, dtype={image.dtype}")

        predictions = model.predict(image)
        print(f"📊 Prediction vector: {predictions}")

        confidence_score = float(np.max(predictions))
        confidence = round(confidence_score * 100, 2)

        # Reject if confidence is too low
        if confidence_score < 0.6:
            return JSONResponse(status_code=400, content={"error": "The uploaded image is not a valid rice plant image or cannot be confidently classified."})

        predicted_index = np.argmax(predictions)
        predicted_label = class_names[predicted_index]

        print(f"✅ Predicted: {predicted_label} ({confidence}%)")

        disease_info = descriptions.get(predicted_label, {
            "description": "No description available.",
            "treatment": ["No treatment available."]
        })

        return {
            "disease": predicted_label,
            "confidence": confidence,
            "description": disease_info["description"],
            "treatment": disease_info["treatment"]
        }

    except Exception as e:
        print(f"❌ Server error: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Internal server error: {str(e)}"})

# from fastapi import FastAPI, File, UploadFile, Header, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# import numpy as np
# import tensorflow as tf
# from PIL import Image
# import io

# app = FastAPI()

# # 🔐 Define your secret API key
# API_KEY = "3d8f7a6c-b913-4f25-a1c7-efaa73a2b9e1"

# # CORS setup (adjust in production)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load your trained model
# model = tf.keras.models.load_model("trainedmodel.keras")

# # Match class order with training
# class_names = [
#     "BACTERIAL LEAF BLIGHT",
#     "BACTERIAL PANICAL BLIGHT",
#     "BROWN SPOT",
#     "DEAD HEART",
#     "HEALTHY",
#     "LEAF BLAST",
#     "LEAF SCALD",
#     "NARROW BROWN SPOT",
#     "NECK BLAST",
#     "RICE HISPA",
#     "SHEATH BLIGHT",
#     "TUNGRO"
# ]

# descriptions = {
#     # Define descriptions here...
# }

# # Image preprocessing function (no normalization)
# def read_imagefile(file_bytes) -> np.ndarray:
#     try:
#         image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
#         image = image.resize((128, 128))  # Match model input size
#         image = np.array(image)
#         image = np.expand_dims(image, axis=0).astype(np.float32)
#         return image
#     except Exception as e:
#         print(f"❌ Error reading image: {e}")
#         return None

# @app.post("/predict/")
# async def predict(
#     file: UploadFile = File(...),
#     x_api_key: str = Header(...),  # 🔐 Require API key in request header
# ):
#     # 🔐 Check if API key is valid
#     if x_api_key != API_KEY:
#         raise HTTPException(status_code=403, detail="Unauthorized: Invalid API Key")

#     try:
#         print(f"📦 Received file: {file.filename}, Content-Type: {file.content_type}")

#         if not file.content_type.startswith("image/"):
#             return JSONResponse(status_code=400, content={"error": "Invalid image file"})

#         image_bytes = await file.read()
#         image = read_imagefile(image_bytes)

#         if image is None:
#             return JSONResponse(status_code=400, content={"error": "Image preprocessing failed"})

#         print(f"🖼️ Image preprocessed: shape={image.shape}, dtype={image.dtype}")

#         predictions = model.predict(image)
#         print(f"📊 Prediction vector: {predictions}")

#         confidence_score = float(np.max(predictions))
#         confidence = round(confidence_score * 100, 2)

#         if confidence_score < 0.6:
#             return JSONResponse(status_code=400, content={"error": "The uploaded image is not a valid rice plant image or cannot be confidently classified."})

#         predicted_index = np.argmax(predictions)
#         predicted_label = class_names[predicted_index]

#         print(f"✅ Predicted: {predicted_label} ({confidence}%)")

#         disease_info = descriptions.get(predicted_label, {
#             "description": "No description available.",
#             "treatment": ["No treatment available."]
#         })

#         return {
#             "disease": predicted_label,
#             "confidence": confidence,
#             "description": disease_info["description"],
#             "treatment": disease_info["treatment"]
#         }

#     except Exception as e:
#         print(f"❌ Server error: {str(e)}")
#         return JSONResponse(status_code=500, content={"error": f"Internal server error: {str(e)}"})
