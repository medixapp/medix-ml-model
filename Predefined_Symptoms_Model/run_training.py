from preprocess_data import*
from model_building import*

def run_training():

    # Get symptoms and diseases data
    df = get_pencernaan_data()

    # Get sampled data
    df_sampled = sample_multiple_n(df, "Gejala", "Penyakit", num_samples=20)

    # Get one hot encoded data
    df_final = one_hot_encode_symptoms(df_sampled, 'Gejala', 'Penyakit')

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
    history = train_model(model, x_train, y_train_encoded, x_test, y_test_encoded)

    # Evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test_encoded)
    print(f'Test accuracy: {accuracy}')

    save_model(model)

if __name__ == "__main__" :
    run_training()