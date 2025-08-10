# Handwritten Text Recognition (IAM) — Web UI + Trained Model

This project implements **offline handwritten text recognition (HTR)** using a model trained on the **IAM Handwriting Database**.  
It includes a **simple web interface** (`index.html`) for uploading a handwriting image, running inference using a **trained model** (provided as a ZIP), and displaying the recognized text.

> ⚠️ The IAM dataset is **not** included. You must obtain it from the official IAM website and follow their license terms.

---

## ✨ Features

- **User-friendly UI** (`index.html`) to upload handwriting images
- **Model inference** using your pre-trained model artifacts
- Displays recognized text instantly
- Works with **word** or **line** crops from IAM (or your own handwriting images with similar preprocessing)

---

## 📂 Project Structure

```
project-root/
├── index.html               # Web interface for file upload
├── assets/
│   ├── style.css            # Optional UI styles
│   └── app.js               # Frontend inference logic or API calls
├── models/
│   ├── trained_model.zip    # Trained model artifacts (uploaded by you)
│   ├── model.onnx           # OR PyTorch/TensorFlow model file after extracting the ZIP
│   ├── vocab.txt            # Character set for decoding predictions
│   └── config.yaml          # Model & preprocessing config
├── backend/                 # Optional (if running inference on server)
│   └── app.py               # Flask API for model inference
└── README.md
```

---

## 🧱 Requirements

### For Frontend-only (ONNX Runtime Web)
- A modern browser (Chrome, Edge, Firefox)
- [onnxruntime-web](https://www.npmjs.com/package/onnxruntime-web) (loaded in `index.html`)

### For Backend API (Flask + PyTorch/TensorFlow)
- Python 3.8+
- Dependencies:
```bash
pip install flask opencv-python numpy torch torchvision  # or tensorflow if using TF model
```

---

## 🔧 Setup

1. **Obtain IAM dataset**  
   Request access and download from the official IAM website.  
   Preprocess images (grayscale, resize height, pad/crop width) exactly as done during training.

2. **Place trained model ZIP**  
   Put your `trained_model.zip` inside the `models/` folder and **extract it** there.  
   This should give you:
   - `model.onnx` **OR** `best_model.pth` / `best_model.h5`
   - `vocab.txt`
   - `config.yaml` (input size, normalization)
   - Any normalization stats file (`mean_std.json`)

3. **Choose inference mode**  
   - **Frontend-only**: use `onnxruntime-web` in `index.html` to run inference in browser  
   - **Backend API**: serve model with Flask and call it from the UI

---

## 🚀 Running the App

### Option A — Frontend-only (ONNX)
```bash
# Start a simple local server
python -m http.server 5500
# Visit:
http://localhost:5500/index.html
```
Upload an image — the recognized text will appear below.

---

### Option B — Backend API (Flask)
```bash
cd backend
python app.py
```
Open `index.html` — it will send the uploaded file to `/infer` on your backend, get the recognized text, and display it.

Example Flask API (`backend/app.py`):
```python
from flask import Flask, request, jsonify
import cv2, numpy as np
import torch

app = Flask(__name__)
# Load model and vocab here

@app.post("/infer")
def infer():
    file = request.files["image"]
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    # preprocess → model inference → decode
    text = run_inference_and_decode(img)  # implement this
    return jsonify({"text": text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

---

## 🔧 Preprocessing (Important)

Your inference must match training preprocessing exactly:

- Convert to **grayscale**
- Resize to a fixed **height** (e.g., 32 or 48px), keep aspect ratio
- Pad or crop width to max width used during training
- Normalize pixel values to `[0,1]` or with training mean/std
- If trained on **word** crops, keep inputs as word images

---

## 🧪 Testing

1. Place a handwriting image (word/line) in the project folder
2. Load it in the UI and run inference
3. Compare output to expected text

---

## 📊 Optional: Evaluation

If you have ground truth text labels:
- Compute **CER** (Character Error Rate)
- Compute **WER** (Word Error Rate)
  
Example:
```
CER = edit_distance(pred_text, gt_text) / len(gt_text)
WER = edit_distance(pred_words, gt_words) / len(gt_words)
```

---

## 🔐 License & Data

- **Code**: Add your preferred license (MIT, Apache 2.0, etc.)
- **Data**: IAM dataset must be obtained from the official site under its license.  
  Do **not** include IAM images in this repo.

---

## 🙌 Acknowledgements

- IAM Handwriting Database  
- ONNX Runtime, Flask, PyTorch, TensorFlow communities  
- Everyone contributing to handwritten text recognition research
