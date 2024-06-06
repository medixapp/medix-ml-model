import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split

def get_data() -> pd.DataFrame :
  """
  Get pandas.DataFrame for diseases data.
  
  Params : None
  
  Return : pandas.DataFrame
  """
  
  try :
    df = pd.read_csv('List Penyakit - Pencernaan_Embeddings.csv')
  except Exception as e :
    print(e)
  
  return df

def str_to_list(dataframe: pd.DataFrame, col: str) -> pd.DataFrame :
  """
  Convert values to list in a column from string with valid list format
  (start and end with [ and ] respectively.)
  
  Params : 
  dataframe : pandas.DataFrame
  col : str -> string of column name
  
  Return : pandas.DataFrame
  """
  
  try :
    dataframe[col] = dataframe[col].apply(lambda x: ast.literal_eval(x))
    if type(dataframe[col].values[0]) == str :
      dataframe[col] = dataframe[col].apply(lambda x: ast.literal_eval(x))
  except Exception as e :
    print(e)
  
  return dataframe

def sample(dataframe: pd.DataFrame, col_of_list: str,
           label_col: str, num_sample: int = 5, n: int = 5,
           random_state: int = 1) -> pd.DataFrame :
  """
  Sample randomly from list for every record. Column col_of_list in the dataframe
  must have list data type as the values.
  
  Params : 
  1. pandas.DataFrame
  
  2. col_of_list (column name) : str -> Column in the dataframe that has list as its values.
  For example : dataframe.loc[0, col_of_list] = [a,b,c]
  
  3. label_col (column name) : str -> Class column in the dataframe.
  
  4. num_sample : int -> How many samples to generate.
  For example, num_samples = 3 and a record has the list [a,b,c]. Then,
  the list will be sampled 3 times, generating new 2 records for the same class.
  
  5. n : int -> How many values for each sample.
  
  6. random_state : int -> Integer for random seed for reproducibility.
  
  Return : pandas.DataFrame
  
  """
  np.random.seed(random_state)
  samples, labels = [], []
  
  try :
    col_of_list_index = dataframe.columns.to_list().index(col_of_list)
    label_col_index = dataframe.columns.to_list().index(label_col)
  
    for record_num in range(len(dataframe)) :
      record_list = dataframe.iloc[record_num, col_of_list_index]
      record_label = dataframe.iloc[record_num, label_col_index]
      if len(record_list) > n :
        for _ in range(num_sample):
          samples.append(np.random.choice(record_list, n, replace=False))
          labels.append(record_label)
      else :
        for _ in range(num_sample):
          samples.append(np.random.choice(record_list, len(record_list)-1, replace=False))
          labels.append(record_label)

    new_df = pd.DataFrame(list(zip(samples, labels)), columns = dataframe.columns)
    return new_df
  
  except Exception as e :
    print(e)

def make_merged_data(dataframe: pd.DataFrame, col_of_list: str,
                     label_col: str, num_samples: list,
                     n_per_samples: list, random_state: int = 1) -> pd.DataFrame :
  """
  Make a pandas DataFrame that contains concatenated pandas DataFrames that have been
  sampled.
  
  Params :
  1. dataframe : pandas.DataFrame
  
  2. col_of_list (column name) : str -> Column in the dataframe that has list as its values.
  For example : dataframe.loc[0, col_of_list] = [a,b,c]
  
  3. label_col (column name) : str -> Class column in the dataframe.
  
  4. num_samples : list -> List consists of number of samples that wanted to be generated.
  For example, [5,4,3] means that the function will return pandas.DataFrame that is the concatenated
  of 3 pandas.DataFrame, upsampled 5, 4, and 3 respectively.
  
  5. n_per_samples : list -> List consists of number of values for each sample in corresponding
  pandas.DataFrame that has been upsampled based on num_samples. For example, if num_samples = [5,4,3]
  and n_per_samples = [4,3,2], this means that the function will return the concatenated pandas.DataFrame
  which consists of these : 5x upsampled pandas.Dataframe, each sample with 4 values/elements, etc.
  
  6. random_state : int -> Integer for random seed for reproducibility.
  
  Return : pandas.DataFrame -> Concatenated dataframe.
  """
  
  datasets = []
  
  try :
    for num_sample, n in zip(num_samples, n_per_samples):
      df_sampled = sample(dataframe, col_of_list, label_col, num_sample, n, random_state)
      datasets.append(df_sampled)
    df_concat = pd.concat(datasets).sort_values(by=label_col).reset_index(drop=True)
    return df_concat
  except Exception as e :
    print(e)

def shuffle(dataframe: pd.DataFrame, random_state: int = 1) -> pd.DataFrame :
  """
  Shuffle the pandas.DataFrame
  
  Params :
  
  1. dataframe : pandas.DataFrame
  2. random_state : int -> Random seed for reproducibility.
  
  Return : pandas.DataFrame -> Shuffled pandas.DataFrame.
  
  """
  
  try :
    new_df = dataframe.sample(len(dataframe), random_state=random_state)
    new_df = new_df.reset_index(drop=True)
    return new_df
  except Exception as e :
    print(e)

def give_numerical_label(dataframe: pd.DataFrame, labels_col: str) -> pd.DataFrame :
  """
  Give numerical labels for the class column
  
  Params :
  
  1. dataframe : pd.DataFrame
  2. labels_col : str -> name of label column.
  
  Return : pandas.DataFrame -> Annotated pandas.DataFrame
  """
  
  try:
    labels = dataframe[labels_col].unique()
    col_dict = {key:value for key, value in zip(list(range(len(labels))), labels)}
    reverse_dict = {value:key for key, value in col_dict.items()}
    dataframe["Label"] = dataframe[labels_col].apply(lambda x: reverse_dict[x])
    return dataframe, col_dict
  except Exception as e :
    print(e)

def split(dataframe: pd.DataFrame, stratify_col: str,
          test_size: float = 0.2, random_state: int = 1) -> tuple[pd.DataFrame, pd.DataFrame] :
  """
  Split the DataFrame.
  
  Params :
  1. dataframe : pandas.DataFrame
  2. stratify_col : str -> Column name to be stratified in train_test_split.
  3. test_size : float -> Percentage of test set in floating point.
  4. random_state : int -> Random seed for reproducibility.
  
  Return : (pandas.DataFrame, pandas.DataFrame) -> train set and test set.
  """
  
  try:
    dataframe_train, dataframe_test = train_test_split(dataframe, test_size=test_size,
                                                       random_state=random_state, stratify=dataframe[stratify_col])
    return dataframe_train, dataframe_test
  except Exception as e:
    print(e)

def list_to_sentences(dataframe: pd.DataFrame, col_of_list: str) -> pd.DataFrame :
  """
  Convert list values in a column of the DataFrame to string.
  
  Params :
  1. dataframe : pandas.DataFrame
  2. col_of_list : str -> Column name that has list data type for its values.
  
  Return : pandas.DataFrame
  """
  
  try :
    dataframe[col_of_list] = dataframe[col_of_list].apply(lambda x: ", ".join(x))
    return dataframe
  except Exception as e :
    print(e)