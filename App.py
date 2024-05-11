import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your trained model
model = tf.keras.models.load_model("model.h5")

# Define the classes for label decoding
class_names = ['Water', 'Land', 'Road', 'Building', 'Vegetation', 'Unlabeled']

# Function to preprocess and predict on user-uploaded image
def predict(image):
    # Preprocess the image
    image = np.array(image.resize((256, 256))) / 255.0
    image = np.expand_dims(image, axis=0)
    # Make prediction
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=3)[0]
    return predicted_class

# Streamlit app layout
st.title('Semantic Segmentation with Deep Learning')

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Make prediction
    predicted_class = predict(image)

    # Display the predicted segmentation mask
    st.image(predicted_class, caption='Predicted Segmentation Mask.', use_column_width=True)
    st.write("")
    st.write("Class Names:")
    st.write(class_names)
