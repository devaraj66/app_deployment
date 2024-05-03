import streamlit as st
import tensorflow as tf
import numpy as np
from keras.preprocessing.image import img_to_array, load_img

# Load your trained model
model = tf.keras.models.load_model('model.h5')

# Define function to make predictions
def predict(image):
    img = load_img(image, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return prediction

# Streamlit app
def main():
    st.title('Image Segmentation App')

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png'])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

        # Make prediction when 'Predict' button is clicked
        if st.button('Predict'):
            with st.spinner('Predicting...'):
                prediction = predict(uploaded_file)
                st.image(prediction, caption='Predicted Mask', use_column_width=True)

if __name__ == '__main__':
    main()
