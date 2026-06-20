from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from fastapi import Form
from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.cm as cm
from fastapi.responses import FileResponse
import tempfile
import os

app = FastAPI(
    title="BrainSight AI",
    description="Brain Tumor Detection API",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
# model = load_model("../models/brain_tumor_multiclass.keras")
model = None
print("BUILT:", model.built)

try:
    print("INPUT:", model.input)
except Exception as e:
    print("INPUT ERROR:", e)

try:
    print("OUTPUT:", model.output)
except Exception as e:
    print("OUTPUT ERROR:", e)

model.build((None, 224, 224, 3))
# Class Names
classes = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]
def generate_gradcam(img_array, model):

    print("MODEL TYPE:", type(model))
    print("FIRST LAYER:", model.layers[0])

    base_model = model.layers[0]

    print("BASE INPUT:", base_model.input)
    print("BASE OUTPUT:", base_model.output)

    last_conv_layer = base_model.get_layer("out_relu")

    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[
            last_conv_layer.output,
            model(base_model.input)
        ]
    )

    img_tensor = tf.convert_to_tensor(
        img_array,
        dtype=tf.float32
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            img_tensor
        )

        pred_index = tf.argmax(
            predictions[0]
        )

        class_channel = predictions[
            :, pred_index
        ]

    grads = tape.gradient(
        class_channel,
        conv_outputs
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    heatmap /= (
        tf.reduce_max(heatmap) + 1e-10
    )

    return heatmap.numpy()

@app.get("/")
def home():
    return {
        "message": "BrainSight AI Backend Running Successfully"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Read Image
    image_bytes = await file.read()

    nparr = np.frombuffer(image_bytes, np.uint8)

    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img = cv2.resize(img, (224, 224))

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)

    predicted_class = np.argmax(prediction)

    result = classes[predicted_class]

    confidence = float(np.max(prediction))

    # Debug Output
    print("\n========== MODEL OUTPUT ==========")

    for i, cls in enumerate(classes):
        print(f"{cls}: {prediction[0][i] * 100:.2f}%")

    print(f"\nFINAL PREDICTION: {result}")
    print(f"CONFIDENCE: {confidence * 100:.2f}%")
    print("==================================\n")

    return {
        "prediction": result,
        "confidence": confidence,
        "probabilities": {
            "Glioma": float(prediction[0][0]),
            "Meningioma": float(prediction[0][1]),
            "No Tumor": float(prediction[0][2]),
            "Pituitary": float(prediction[0][3])
        }
    }

@app.post("/gradcam")
async def gradcam(file: UploadFile = File(...)):

    image_bytes = await file.read()

    nparr = np.frombuffer(image_bytes, np.uint8)

    original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img = cv2.resize(original_img, (224, 224))

    img = img / 255.0

    img_array = np.expand_dims(img, axis=0)

    heatmap = generate_gradcam(img_array, model)

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.resize(
        heatmap,
        (original_img.shape[1], original_img.shape[0])
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    superimposed_img = cv2.addWeighted(
        original_img,
        0.6,
        heatmap,
        0.4,
        0
    )

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    )

    cv2.imwrite(
        temp_file.name,
        superimposed_img
    )

    return FileResponse(
        temp_file.name,
        media_type="image/jpeg"
    )

@app.post("/generate-report")
async def generate_report(
    prediction: str = Form(...),
    confidence: float = Form(...),
    patient_name: str = Form(...),
    age: str = Form(...),
    gender: str = Form(...)
):

    pdf_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    )

    doc = SimpleDocTemplate(pdf_file.name)

    styles = getSampleStyleSheet()

    content = []

    # Title
    content.append(
        Paragraph(
            "BrainSight AI Medical Report",
            styles["Title"]
        )
    )

    content.append(Spacer(1, 20))

    # Patient Details
    content.append(
        Paragraph(
            f"<b>Patient Name:</b> {patient_name}",
            styles["Normal"]
        )
    )

    content.append(
        Paragraph(
            f"<b>Age:</b> {age}",
            styles["Normal"]
        )
    )

    content.append(
        Paragraph(
            f"<b>Gender:</b> {gender}",
            styles["Normal"]
        )
    )

    content.append(Spacer(1, 20))

    # Diagnosis
    content.append(
        Paragraph(
            f"<b>Diagnosis:</b> {prediction}",
            styles["Normal"]
        )
    )

    content.append(
        Paragraph(
            f"<b>Confidence Score:</b> {confidence:.2f}%",
            styles["Normal"]
        )
    )

    content.append(Spacer(1, 20))

    # AI Summary
    content.append(
        Paragraph(
            """
            <b>AI Clinical Summary</b><br/><br/>
            This MRI scan was analyzed using BrainSight AI.
            The model identified imaging patterns associated
            with the predicted class and generated a confidence
            score based on deep learning inference.
            """,
            styles["Normal"]
        )
    )

    content.append(Spacer(1, 20))

    # Recommendation
    content.append(
        Paragraph(
            """
            <b>Recommendation</b><br/><br/>
            Please consult a qualified neurologist,
            radiologist, or healthcare professional
            for further medical evaluation.
            """,
            styles["Normal"]
        )
    )

    content.append(Spacer(1, 20))

    # Disclaimer
    content.append(
        Paragraph(
            """
            <b>Disclaimer:</b>
            This AI system is intended for educational
            and research purposes only and should not be
            considered a substitute for professional
            medical diagnosis.
            """,
            styles["Italic"]
        )
    )

    doc.build(content)

    return FileResponse(
        pdf_file.name,
        filename="BrainSight_Report.pdf",
        media_type="application/pdf"
    )