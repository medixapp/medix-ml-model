import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import json

model = tf.keras.models.load_model('predefined_model.h5')
label_encoder = LabelEncoder()

with open('all_symptoms.txt', 'r') as symptoms_file:
    all_symptoms = symptoms_file.read().split(',')

with open('class_dict.json', 'r') as class_json:
    class_dict = json.load(class_json)

unique_classes = list(class_dict.values())
label_encoder.fit(unique_classes)

def preprocess_input(input_symptoms, all_symptoms):
    # Remove any extra features not present in all_symptoms
    input_symptoms = [symptom for symptom in input_symptoms if symptom in all_symptoms]

    # Ensure the input data has the same number of features as the model expects
    input_data = [int(symptom in input_symptoms) for symptom in all_symptoms[:58]]

    return input_data

def predict_and_display(input_data, model, label_encoder):
    # Make predictions
    predictions = model.predict([input_data])

    # Get the index of the highest probability prediction
    max_prob_index = np.argmax(predictions)

    # Decode the predicted class from the index using the label_encoder
    predicted_class = label_encoder.inverse_transform([max_prob_index])[0]

    # Get the probability corresponding to the highest prediction
    max_prob = predictions[0][max_prob_index]

    # Display the predicted class and its probability
    print(f"Predicted Class: {predicted_class}, Probability: {max_prob}")

if __name__ == "__main__":
    # Get input symptoms from the user
    input_symptoms_str = input("Masukkan gejala yang dirasakan: ")
    input_symptoms = [symptom.strip() for symptom in input_symptoms_str.split(',')]

    # Preprocess input data and make predictions
    input_data = preprocess_input(input_symptoms, all_symptoms)
    predict_and_display(input_data, model, label_encoder)

