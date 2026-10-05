import os
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO

# Create Flask application
app = Flask(__name__)

# Folder for uploaded images
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load your trained mango disease model
model = YOLO("best.pt")
disease_info = {
    "Anthracnose": {
        "description": "Anthracnose is a fungal disease that can affect mango leaves and fruits.",
        "prevention": "Remove affected plant parts and maintain good air circulation."
    },

    "Powdery_Mildew": {
        "description": "Powdery mildew is a fungal disease that can appear as a white powdery growth on mango plant parts.",
        "prevention": "Maintain proper spacing and good air circulation around the plant."
    },

    "Cutting_Weevil": {
        "description": "Cutting weevil is an insect-related pest problem that can damage mango plant parts.",
        "prevention": "Regularly inspect the plant and remove affected parts."
    },

    "Healthy": {
        "description": "The uploaded image is classified as healthy by the model.",
        "prevention": "Continue regular monitoring and good mango plant care."
    },

    "Bacterial_Canker": {
        "description": "Bacterial canker is a bacterial disease that can cause lesions and damage to mango plant parts.",
        "prevention": "Remove affected parts and maintain good orchard hygiene."
    },

    "Sooty_Mould": {
        "description": "Sooty mould appears as a dark coating on plant surfaces.",
        "prevention": "Monitor and control insect pests that may contribute to sooty mould."
    },

    "Die_Back": {
        "description": "Die-back causes drying and death of shoots or branches of the mango plant.",
        "prevention": "Prune affected branches and maintain proper plant care."
    },

    "Gall_Midge": {
        "description": "Gall midge is an insect pest that can cause abnormal growth or damage to mango plant parts.",
        "prevention": "Regularly inspect new growth and manage affected plant parts."
    }
}


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    filename = None
    description = ""
    prevention = ""

    if request.method == "POST":
        if "image" not in request.files:
            return render_template("index.html")

        file = request.files["image"]

        if file.filename == "":
            return render_template("index.html")

        filename = secure_filename(file.filename)
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(image_path)

        results = model(image_path)
        result = results[0]

        top1_index = result.probs.top1
        prediction = result.names[top1_index]
        confidence = round(float(result.probs.top1conf) * 100, 2)

        info = disease_info.get(prediction, {})
        description = info.get("description", "")
        prevention = info.get("prevention", "")

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        filename=filename,
        description=description,
        prevention=prevention
    )
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory("uploads", filename)


if __name__ == "__main__":
    app.run(debug=True)