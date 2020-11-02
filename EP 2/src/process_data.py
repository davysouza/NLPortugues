import pandas as pd
import numpy as np

import tensorflow as tf

from tensorflow import data

from tensorflow import keras
from tensorflow.keras import layers

def clean_data(dataset, selected_colums):
    '''
        Select the specified columns, drop all samples that has NaN values and
        return the cleaned dataset

        Parameters: 
            dataset (pandas DataFrame): The dataset to be cleaned

        Returns:
            The dataset cleaned
    '''
    # selecting columns
    processed_data = dataset[selected_colums]

    # dropping NaN values
    processed_data = processed_data.dropna()

    return processed_data

def shuffle_and_split(dataset):
    '''
        Shuffle data, splits dataset into training, validation and test

        Parameters: 
            dataset (pandas DataFrame): Dataset that will be shuffled and splitted

        Returns:
            training_dataset, validation_dataset and testing_dataset
    '''
    # shuffling data
    processed_data = dataset.sample(frac=1).reset_index(drop=True)

    # splitting data
    validation_data_offset = int(.65 * len(processed_data))
    testing_data_offset = int(.75 * len(processed_data))
    
    return np.split(processed_data, [validation_data_offset, testing_data_offset])

def create_vectorizer(dataset, max_tokens=20000, output_len=50, batch_size=128):
    '''
        Create a vectorizer using TextVectorization layer. 

        Parameters: 
            dataset: Data to be vectorized
            max_tokens (int): maximum of tokens that will be considered. Default is 20k,
                              which means that the 20k top words will be considered
            output_len (int): output length. All sequences will be padded or trucated
                              to reaches output length. Default is 50
            batch_size (int): Size of a batch. Default is 128

        Returns:
            vectorizer (obj)
    '''
    vectorizer = layers.experimental.preprocessing.TextVectorization(max_tokens=max_tokens, output_sequence_length=output_len)
    text_ds = data.Dataset.from_tensor_slices(dataset).batch(batch_size)
    vectorizer.adapt(text_ds)

    return vectorizer

def vectorize_data(vectorizer, x_data, y_data):
    '''
        Vectorize data to be used on model

        Parameters: 
            vectorizer (obj): Vectorizer object
            x_data (int): text that will be vectorized
            y_data (int): labels

        Returns:
            vectorized_data
    '''
    x_train_embedded = vectorizer(np.array([[s] for s in x_data])).numpy()
    y_train_embedded = np.array(y_data)

    return x_train_embedded, y_train_embedded


