from model_embeddings_helper import *
import tensorflow as tf
import json

def run_training():
    train_data, test_data, col_dict = get_train_test_data("Gejala", "Penyakit",
                                                        [30]*6, [8,7,6,5,4,3])

    train_sequenced, test_sequenced, word_index = get_train_test_sequences((train_data, test_data))

    model = build_model()

    compile_model(model, tf.keras.optimizers.Adam(),
                tf.keras.losses.SparseCategoricalCrossentropy(),
                ['accuracy'])

    history = training(model, (train_sequenced, test_sequenced))
    print("\nTraining accuracy = {:.2f} %\nTesting accuracy = {:.2f} %".format(history.history['accuracy'][-1]*100,
                                                                               history.history['val_accuracy'][-1]*100))
    
    save_model(model)
    
    with open('word_index.json', 'w') as words:
        json.dump(word_index, words)
        
    with open('label_dict.json', 'w') as labels_dict:
        json.dump(col_dict, labels_dict)

if __name__ == "__main__" :
    run_training()