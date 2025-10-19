#Bonus Task: Deployment with Streamlit
# Libraries
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Load the trained model
model = tf.keras.models.load_model('my_mnist_model.h5') # Save the model

st.title("MNIST Digit Classifier")
st.write("Upload an image of a handwritten digit (0-9)")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L') # Open and convert to grayscale
    image = np.array(image)
    
    # Preprocess the image to look like MNIST data
    img_resized = cv2.resize(image, (28, 28))
    # Invert colors if background is black (MNIST has white background)
    if img_resized.mean() > 127: 
        img_resized = 255 - img_resized
    img_normalized = img_resized.astype("float32") / 255.0
    img_final = np.expand_dims(img_normalized, axis=(0, -1)) # Shape (1, 28, 28, 1)
    
    # Make prediction
    prediction = model.predict(img_final)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction)
    
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write(f"**Prediction: {predicted_digit}**")
    st.write(f"**Confidence: {confidence:.2f}**")