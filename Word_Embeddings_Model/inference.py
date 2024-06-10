import tensorflow as tf
import re
import json
import numpy as np

def remove_punc(string: str) -> str:
    """
    Remove punctuations from string.
    
    Param : 
    
    1. string : str
    
    Return : str
    """
    new_str = re.sub(r'[^\w\s]', ' ', string)
    return new_str

def get_model() -> tf.keras.models.Model :
    """
    Load the model.
    
    Return : tf.keras.models.Model
    """
    model = tf.keras.models.load_model('model.h5')
    return model

def get_word_index() -> dict:
    """
    Get the word index.
    
    Return : dict -> The word index.
    """
    with open('word_index.json', 'r') as word_index :
        words = json.load(word_index)
    return words

def get_label_dict() -> dict:
    """
    Get the label dictionary to translate the prediction from integer to string.
    
    Return : dict -> The label dictionary.
    """
    with open('label_dict.json', 'r') as labels :
        label_dict = json.load(labels)
    new_label_dict = {}
    for key, value in label_dict.items() :
        new_label_dict[int(key)] = value
    return new_label_dict

def to_sequence(string: str, word_index: dict,
                max_length: int = 88) -> np.ndarray :
    """
    Convert the sentence into sequence of integers, refers from the word index.
    
    Params :
    
    1. string : str -> The sentence to be converted.
    
    2. word_index : dict -> The word index dictionary.
    
    3. max_length : int -> The maximum length of the sequences. Must match the max_length for the model.
    
    Return : np.ndarray -> The sequence in numpy array form.
    """
    sentence = remove_punc(string).lower()
    sentence_arr = sentence.split()
    words = word_index.keys()
    sequence = []
    for w in sentence_arr :
        if w in words :
            sequence.append(word_index[w])
        else :
            sequence.append(1)
    
    if len(sequence) < max_length :
        num_zero = max_length - len(sequence)
        sequence += [0]*num_zero
    else :
        sequence = sequence[:max_length]
        
    sequence = np.array(sequence).reshape((1, -1))
    return sequence
    
def predict(model: tf.keras.models.Model, sequence: np.ndarray, 
            label_dict: dict) -> str :
    """
    Predict the class using the model.
    
    Params :
    
    1. model : tf.keras.models.Model
    
    2. sequence : np.ndarray -> The sequence of integers, resulted from the conversion based on the word index.
    
    3. label_dict : dict -> The label dictionary that stores integers as its keys and the class string as its values.
    
    Return : str -> The predicted class.
    """
    result = model.predict(sequence)
    class_pred = np.argmax(result)
    prediction = label_dict[class_pred]
    
    return prediction

def main() :
    sentence = input("\nMasukkan gejala Anda : ").strip()
    
    model = get_model()
    word_index = get_word_index()
    label_dict = get_label_dict()
    
    sequence = to_sequence(sentence, word_index)
    pred = predict(model, sequence, label_dict)
    
    print("\nPrediction : {}".format(pred))

if __name__ == "__main__" :
    main()