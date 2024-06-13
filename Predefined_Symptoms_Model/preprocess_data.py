import pandas as pd
import numpy as np
import ast
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

def get_pencernaan_data():
    """
    Get pandas.DataFrame for disease and symptoms.

    Params: None

    Return: pandas.DataFrame
    """
    try:
        df = pd.read_csv('list_penyakit.csv')
    except Exception as e:
        print(e)
        df = pd.DataFrame()  # Return an empty DataFrame in case of an error

    return df


def sample_data(dataframe, col_of_list, label_col, num_samples=5, n=5):
    """
    Sample data from a DataFrame where one column contains lists and another contains labels.

    Params:
    - dataframe (pd.DataFrame): The input DataFrame.
    - col_of_list (str): The column name which contains lists.
    - label_col (str): The column name which contains labels.
    - num_samples (int): Number of samples to generate for each record.
    - n (int): Number of elements to sample from each list.

    Returns:
    - pd.DataFrame: A new DataFrame with sampled data.
    """
    samples, labels = [], []
    col_of_list_index = dataframe.columns.to_list().index(col_of_list)
    label_col_index = dataframe.columns.to_list().index(label_col)

    for record_num in range(len(dataframe)):
        record_list = dataframe.iloc[record_num, col_of_list_index]
        record_label = dataframe.iloc[record_num, label_col_index]

        # Ensure record_list is a list
        if isinstance(record_list, str):
            try:
                record_list = ast.literal_eval(record_list)
            except (ValueError, SyntaxError):
                record_list = record_list.split(', ')
        elif not isinstance(record_list, list):
            record_list = list(record_list)

        if len(record_list) >= n:
            for _ in range(num_samples):
                samples.append(np.random.choice(record_list, n, replace=False).tolist())
                labels.append(record_label)
        else:
            for _ in range(num_samples):
                samples.append(np.random.choice(record_list, len(record_list), replace=False).tolist())
                labels.append(record_label)

    new_df = pd.DataFrame(list(zip(samples, labels)), columns=[col_of_list, label_col])
    return new_df


def sample_multiple_n(dataframe, col_of_list, label_col, num_samples=20, n_values=[5, 4, 3, 2]):
    """
    Create multiple sampled DataFrames for different values of n and concatenate them.

    Params:
    - dataframe (pd.DataFrame): The input DataFrame.
    - col_of_list (str): The column name which contains lists.
    - label_col (str): The column name which contains labels.
    - num_samples (int): Number of samples to generate for each record.
    - n_values (list of int): List of n values to use for sampling.

    Returns:
    - pd.DataFrame: A concatenated DataFrame with all samples.
    """
    sampled_dfs = [dataframe]

    for n in n_values:
        sampled_df = sample_data(dataframe, col_of_list, label_col, num_samples, n)
        sampled_dfs.append(sampled_df)

    concatenated_df = pd.concat(sampled_dfs).sort_values(by=[label_col]).reset_index(drop=True)
    return concatenated_df

def one_hot_encode_symptoms(dataframe, symptoms_col, label_col):
    """
    One-hot encoding the symptoms in the specified column and concatenate the result with the label column.

    Params:
    - dataframe (pd.DataFrame): The input DataFrame.
    - symptoms_col (str): The column name containing the symptoms lists.
    - label_col (str): The column name containing the labels.

    Returns:
    - pd.DataFrame: A DataFrame with one-hot encoded symptoms with the original labels.
    """
    # Ensure symptoms elements are lists
    dataframe[symptoms_col] = dataframe[symptoms_col].apply(
        lambda x: x if isinstance(x, list) else x.tolist() if isinstance(x, np.ndarray) else x.split(', ')
    )

    # Get all unique symptoms
    all_symptoms = sorted(list(set(sum(dataframe[symptoms_col], []))))

    # Create a binary matrix for one-hot encoding
    binary_matrix = {symptom: dataframe[symptoms_col].apply(lambda x: int(symptom in x)) for symptom in all_symptoms}

    # Create a one-hot encoded DataFrame
    one_hot_encoded_df = pd.DataFrame(binary_matrix)

    # Concatenate the label column with the one-hot encoded DataFrame
    df_final = pd.concat([dataframe[label_col], one_hot_encoded_df], axis=1)

    return df_final


def shuffle_and_split(dataframe, label_col, test_size=0.2, random_state=42):
    """
    Shuffle the DataFrame and split it into train and test sets, ensuring that each class in the label column
    exist in both sets.

    Params:
    - dataframe (pd.DataFrame): The input DataFrame.
    - label_col (str): The column name containing the class labels.
    - test_size (float): The proportion of the dataset to include in the test split.
    - random_state (int): Random seed for reproducibility.

    Returns:
    - df_train (pd.DataFrame): The training set.
    - df_test (pd.DataFrame): The test set.
    """
    # Shuffle the DataFrame
    df_shuffled = dataframe.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Split the data into train and test sets
    df_train, df_test = train_test_split(df_shuffled, test_size=test_size, stratify=df_shuffled[label_col],
                                         random_state=random_state)

    return df_train, df_test

def extract_features_and_labels(df_train, df_test, label_col):
    """
    Extract features (x) and labels (y) from training and testing DataFrames.

    Params:
    - df_train (pd.DataFrame): The training DataFrame.
    - df_test (pd.DataFrame): The testing DataFrame.
    - label_col (str): The column name containing the labels.

    Returns:
    - x_train (np.ndarray): Training features.
    - y_train (pd.Series): Training labels.
    - x_test (np.ndarray): Testing features.
    - y_test (pd.Series): Testing labels.
    """
    # Extract features and labels
    x_train, y_train = df_train.drop(label_col, axis=1), df_train[label_col]
    x_test, y_test = df_test.drop(label_col, axis=1), df_test[label_col]

    # Convert features to int32
    x_train, x_test = x_train.astype('int32'), x_test.astype('int32')

    # Convert features to numpy arrays
    x_train, x_test = x_train.values, x_test.values

    return x_train, y_train, x_test, y_test


def encode_labels(y_train, y_test):
    """
    Encode and one-hot encode labels for training and testing sets.

    Params:
    - y_train (pd.Series or np.ndarray): Training labels.
    - y_test (pd.Series or np.ndarray): Testing labels.

    Returns:
    - y_train_encoded (np.ndarray): Encoded and one-hot encoded training labels.
    - y_test_encoded (np.ndarray): Encoded and one-hot encoded testing labels.
    - label_encoder (LabelEncoder): Fitted LabelEncoder instance.
    """
    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    # Fit and transform the training labels
    y_train_encoded = label_encoder.fit_transform(y_train)

    # Transform the testing labels
    y_test_encoded = label_encoder.transform(y_test)

    # One-hot encode the labels
    y_train_encoded = tf.cast(to_categorical(y_train_encoded), tf.int32).numpy()
    y_test_encoded = tf.cast(to_categorical(y_test_encoded), tf.int32).numpy()

    return y_train_encoded, y_test_encoded, label_encoder
