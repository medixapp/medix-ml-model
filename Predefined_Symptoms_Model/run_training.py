import json
import tensorflow as tf
from preprocess_data import*
from model_building import*

def run_training():

    # Get symptoms and diseases data
    df = get_pencernaan_data()

    # Get sampled data
    df_sampled = sample_multiple_n(df, "Gejala", "Penyakit", num_samples=20)

    # Get one hot encoded data
    df_final, all_symptoms = one_hot_encode_symptoms(df_sampled, 'Gejala', 'Penyakit')

    # Get training and testing set
    df_train, df_test = shuffle_and_split(df_final, 'Penyakit', test_size=0.2)

    # Get extracted feature and label of training and testing data
    x_train, y_train, x_test, y_test = extract_features_and_labels(df_train, df_test, 'Penyakit')

    # Encode labels
    y_train_encoded, y_test_encoded, label_encoder = encode_labels(y_train, y_test)

    # Determine input shape and number of classes
    num_classes = y_train_encoded.shape[1]

    # Build the model
    model = build_model(num_classes)

    # Compile the model
    model = compile_model(model)

    # Train the model
    _ = train_model(model, x_train, y_train_encoded, x_test, y_test_encoded)

    # Evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test_encoded)
    print(f'Test accuracy: {accuracy:.2f}')
    print(f'Test loss: {loss:.2f}')

    save_model(model)

    with open('all_symptoms.txt', 'w') as txt_file:
        txt_file.write(",".join(all_symptoms))

    class_dict = {i: label_encoder.inverse_transform([i])[0] for i in range(14)}
    with open('class_dict.json', 'w') as json_file:
        json.dump(class_dict, json_file)

if __name__ == "__main__" :
    run_training()
