# Apps/classification.py
import streamlit as st
import pickle
from PIL import Image
import numpy as np
from keras.models import load_model, Model
from keras.applications.vgg16 import preprocess_input
from sklearn.svm import SVC
from keras import backend as K

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

# Load the SVM model
model_svm_rbf_path = r"D:\Users\RESA\Coding\Evaluasi\VGG16_svm_model_rbf.pkl"
model_svm_rbf = pickle.load(open(model_svm_rbf_path, 'rb'))

# Load the pre-trained VGG16 model with custom objects
model_path = r"D:\Users\RESA\Coding\Evaluasi\VGG16.h5"
loaded_model = load_model(model_path, custom_objects={'specificity': specificity, 'sensitivity': sensitivity})

# Extract features from the last layer of VGG16
extractCNN = Model(loaded_model.inputs, loaded_model.layers[-2].output)

# Define class mapping
class_mapping = {
    0: 'CN',
    1: 'EMCI',
    2: 'LMCI',
    3: 'AD'
}

# Function to preprocess the image for VGG16 model
def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Function to extract VGG16 features
def extract_vgg16_features(img_array):
    features = extractCNN.predict(img_array)
    features = features.flatten()
    return features

# Function to perform classification with class mapping
def classify_image(img_array):
    # Extract VGG16 features
    vgg16_features = extract_vgg16_features(img_array)

    # Make prediction using the SVM model
    prediction_index = model_svm_rbf.predict(vgg16_features.reshape(1, -1))[0]
    
    # Map the predicted index to the corresponding class label
    predicted_class = class_mapping.get(prediction_index, "Unknown")
    return predicted_class

# Streamlit app
def app():
    st.title("Image Classification with VGG16 and SVM")

    # Upload image directly in the main area
    uploaded_file = st.file_uploader("Choose an image...", type="png")

    if uploaded_file is not None:
        # Display the uploaded image
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image.", use_column_width=True)

        # Preprocess the image
        img_array = preprocess_image(uploaded_file)

        # Perform classification
        predicted_class = classify_image(img_array)

        # Display the result
        st.write("Prediction:", predicted_class)

if __name__ == "__main__":
    app()
