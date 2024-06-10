import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from data_processing import *

def get_train_test_data(col_of_list: str,
                        label_col: str, num_samples: list,
                        n_per_samples: list,
                        random_state: int = 1) -> tuple[tuple, tuple, dict] :
    """
    Get train and test data.
    
    Params :
    
    1. col_of_list : str -> Column name that has list data type for its values.
    
    2. label_col : str -> Name of label column.
    
    3. num_samples : list -> List consists of number of samples that wanted to be generated.
    For example, [5,4,3] means that the function will return pandas.DataFrame that is the concatenated
    of 3 pandas.DataFrame, upsampled 5, 4, and 3 respectively.
    
    4. n_per_samples : list -> List consists of number of values for each sample in corresponding
    pandas.DataFrame that has been upsampled based on num_samples. For example, if num_samples = [5,4,3]
    and n_per_samples = [4,3,2], this means that the function will return the concatenated pandas.DataFrame 
    which consists of these : 5x upsampled pandas.Dataframe, each sample with 4 values/elements, etc.
    
    5. random_state : int -> Integer for random seed for reproducibility.
  
    Return : 
    
    tuple -> Consists of 3 elements :
    
    1. train_data : tuple of train sentences and train labels,
    
    2. test_data : tuple of test sentences and test labels, 
    
    3. col_dict : dict -> Dictionary that saves labels in integer as the key and the real labels as the values.
    """
    df = get_data()
    df = str_to_list(df, col_of_list)
    df = make_merged_data(df, col_of_list, label_col,
                        num_samples, n_per_samples, random_state)
    df = shuffle(df, random_state)
    df, col_dict = give_numerical_label(df, label_col)
    df = list_to_sentences(df, col_of_list)
    train_df, test_df = split(df, "Label")
    
    train_sentences = train_df[col_of_list].values
    test_sentences = test_df[col_of_list].values
    train_labels = train_df["Label"].values
    test_labels = test_df["Label"].values
    
    train_data = (train_sentences, train_labels)
    test_data = (test_sentences, test_labels)
    
    return train_data, test_data, col_dict

def get_train_test_sequences(data: tuple, vocab_size: int = 1000,
                             max_length: int = 88, oov_tok: str = "<UNK>",
                             padding_type: str = "post", trunc_type: str = "post") -> tuple[tuple, tuple, dict] :
    """
    Get training and testing sequences with their labels.
    
    Params :
    
    1. data : tuple -> Consists of train_data and test_data, each is tuple with 2 elements, 
    the sentences and the labels.
    
    2. vocab_size : int -> Maximum number of words the tokenizer could save.
    
    3. max_length : int -> Maximum length of a sentence that the tokenizer could save.
    
    4. oov_tok : str -> Out-of-vocab token for unseen words in training data.
    
    5. padding_type : str -> Type of padding in pad_sequences.
    
    6. trunc_type : str -> Type of truncating in pad_sequences.
    
    Return : 
    
    tuple -> Consists of 3 elements :
    
    1. train_data : tuple of train sequences and train labels,
    
    2. test_data : tuple of test sequences and test labels
    
    3. word_index : dict -> Dictionary that saves word index from the tokenizer.
    """
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    (train_sentences, train_labels), (test_sentences, test_labels) = data
    tokenizer.fit_on_texts(train_sentences)
    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    test_sequences = tokenizer.texts_to_sequences(test_sentences)
    
    train_sequences = pad_sequences(train_sequences, maxlen=max_length,
                                    padding=padding_type, truncating=trunc_type)
    test_sequences = pad_sequences(test_sequences, maxlen=max_length,
                                   padding=padding_type, truncating=trunc_type)
    
    train_data = train_sequences, train_labels
    test_data = test_sequences, test_labels
    return train_data, test_data, tokenizer.word_index

def build_model(vocab_size: int = 1000, embedding_dim: int = 32,
                max_length: int = 88, print_summary: bool = False) -> tf.keras.models.Model :
    """
    Building Deep Learning model for text classification.
    
    Params :
    
    1. vocab_size : int -> Maximum number of words the tokenizer could save.
    
    2. embedding_dim : int -> Dimension of word embedding used in Embedding Layer.
    
    3. max_length : int -> Maximum length of a sentence that the tokenizer could save.
    
    4. print_summary : bool -> Print model summary if True.
    
    Return : tf.keras.models.Model
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Conv1D(filters=8, kernel_size=8, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(filters=16, kernel_size=16, activation='relu'),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=14, activation='softmax')
    ])
    
    if print_summary :
        model.summary()
    return model

def compile_model(model: tf.keras.models.Model,
                  optimizer: tf.keras.optimizers.Optimizer,
                  loss: tf.keras.losses.Loss,
                  metrics: list) :
    """
    Compile the model with optimizer, loss, and metrics.
    
    Params :
    
    1. model : tf.keras.models.Model
    
    2. optimizer : tf.keras.optimizers.Optimizer
    
    3. loss : tf.keras.losses.Loss
    
    4. metrics : list -> Consists of strings that represents metrics' names that 
    should be displayed while training.
    
    Return : None
    """
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)
    
def make_callback(threshold: float = 0.98) -> tf.keras.callbacks.Callback :
    """
    Make custom callback that stop training where the metrics have reached certain threshold.
    
    Params :
    
    1. threshold : float -> threshold of the metrics in floating point percentage.
    
    Return : tf.keras.callbacks.Callback
    """
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs.get('accuracy') >= threshold and logs.get('val_accuracy') >= threshold :
                self.model.stop_training = True
        
    mycallback = myCallback()
    return mycallback

def training(model: tf.keras.models.Model, data: tuple, epochs: int = 1000,
             use_callback: bool = True) -> tf.keras.callbacks.History :
    """
    Training the model.
    
    Params :
    
    1. model : tf.keras.models.Model
    
    2. data : tuple -> Consists of train_data and test_data, each is tuple with 2 elements, 
    the sentences and the labels.
    
    3. epochs : int -> Number of epochs for training.
    
    4. use_callback : bool -> Use the custom callback if True.
    
    Return : tf.keras.callbacks.History
    """
    (train_data, train_labels), validation_data = data
    
    if use_callback :
        callback = make_callback()
        history = model.fit(train_data, train_labels, epochs=epochs,
                            validation_data=validation_data, callbacks=[callback])
    else :
        history = model.fit(train_data, train_labels, epochs=epochs,
                            validation_data=validation_data)
    
    return history

def save_model(model: tf.keras.models.Model) :
    model.save('model.h5')