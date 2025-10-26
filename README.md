# MNIST Digit Classifier (Streamlit App)

A simple Streamlit web app for classifying handwritten digits (0–9) using a TensorFlow/Keras model trained on MNIST.

---

## ✨ Features
- 📤 Upload a PNG/JPG of a handwritten digit
- 🧪 Automatic preprocessing to 28×28 grayscale like MNIST
- 🤖 TensorFlow model inference with predicted digit and confidence
- ⚡ Lightweight CPU-only dependencies for easy deployment

---

## 📁 Project Structure
```
.
├── app.py                 # Streamlit app entry point
├── requirements.txt       # Python dependencies
├── my_mnist_model.h5      # Trained Keras model (required at runtime)
├── Deep_Learning.ipynb    # (Optional) training/experiments – not required to run
├── ML.ipynb               # (Optional) experiments – not required to run
└── NLP_spaCy.ipynb        # (Optional) experiments – not required to run
```
> **Note:** The notebooks are optional and not needed to serve the app.

---

## 📦 Requirements
See `requirements.txt` for exact versions. Core runtime libraries include:
- streamlit
- tensorflow-cpu
- numpy
- pandas
- scikit-learn
- matplotlib
- opencv-python-headless

> If you encounter OpenCV GUI backend errors, keep using `opencv-python-headless` (no GUI).

---

## 🚀 Quickstart

### 1) Create & activate a virtual environment (recommended)
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3) Ensure the model file is present
Place the trained model at the project root as:
```
my_mnist_model.h5
```

### 4) Run the app
```bash
streamlit run app.py
```
The app will open in your browser (usually http://localhost:8501).

---

## 🧠 How Inference Works (at a glance)
1. Load the Keras model from `my_mnist_model.h5`.
2. User uploads an image (PNG/JPG).
3. Image is converted to grayscale, resized to **28×28**, normalized to **[0,1]**, and reshaped to **(1, 28, 28, 1)**.
4. Model predicts a probability distribution over digits **0–9**.
5. App displays the **predicted digit** and **confidence**.

> Tip: If your input digits appear inverted (white digit on dark background or vice‑versa), you can preprocess images in your editor, or adjust the inversion heuristic in `app.py`.

---

## 🧪 Model Notes
- The app expects an MNIST-style model with input shape `(28, 28, 1)` and softmax outputs for 10 classes.
- You can train your own model and save it as `my_mnist_model.h5` (Keras `model.save(...)`).
- For reproducibility, ensure the saved model’s preprocessing matches the app’s preprocessing.

---

## 🛠 Common Issues & Fixes
- **`OSError: No file or directory: 'my_mnist_model.h5'`**  
  Make sure the model file exists at the project root.
- **`ImportError: libcudart.so...`** or GPU-related warnings  
  You’re on CPU-only TensorFlow (`tensorflow-cpu`). These warnings can be ignored.
- **OpenCV errors about display backends**  
  Use `opencv-python-headless` (already in requirements).
- **Bad predictions on certain backgrounds**  
  Try cropping tighter to the digit and using a plain background. If needed, tweak the inversion condition in `app.py`.

---

## 🧪 Local Testing Snippet
If you want to test preprocessing/prediction outside Streamlit:
```python
import numpy as np, cv2, tensorflow as tf
from PIL import Image

model = tf.keras.models.load_model("my_mnist_model.h5")

def predict_image(path):
    image = Image.open(path).convert("L")
    arr = np.array(image)
    arr = cv2.resize(arr, (28, 28))
    # Optional: invert if needed for your data
    # arr = 255 - arr
    arr = (arr.astype("float32") / 255.0)[None, ..., None]
    probs = model.predict(arr)
    return int(np.argmax(probs)), float(np.max(probs))

print(predict_image("sample_digit.png"))
```

---

## 📦 (Optional) Docker
```dockerfile
# Simple CPU image
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```
Build & run:
```bash
docker build -t mnist-streamlit .
docker run -p 8501:8501 mnist-streamlit
```

---

## ☁️ Deployment Ideas
- **Streamlit Community Cloud:** One-click deploy from your repo.
- **Heroku / Fly.io / Railway:** Use the Dockerfile above.
- **K8s**: Containerize with the provided Dockerfile and deploy a service.

---

## 🔐 Security Notes
- This demo accepts arbitrary image uploads. For production, consider size/type checks and rate limiting.
- Avoid bundling sensitive assets or credentials.

---

## 📄 License
MIT (or your preferred license).

---

## 🙌 Acknowledgements
- MNIST dataset by Yann LeCun et al.
- Streamlit & TensorFlow teams for great tooling.
