import numpy as np

import tensorflow as tf

from tensorflow import data

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks

def load_word_embedding(embedding_path):
    '''
        Load a word embedding CSV file into a map of embeddings

        Parameters: 
            embedding_path (str): Path to word embedding file

        Returns:
            embeddings_index (dict(str, array)): map from a character to their 
            word embedding vector
    '''
    embeddings_index = {}
    with open(embedding_path, encoding="utf8") as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs

    return embeddings_index

def create_embedding_matrix(word_index, embeddings_index, num_tokens, embedding_dim=50):
    '''
        Creates a embedding matrix from word index and embeddings index provided.
        Words not found in embadding index will be all-zeros.

        Parameters: 
            word_index (dict(str, int)): word index. A map from a word to ther index
            embeddings_index (dict(str, array)): A map from a word to their embedding vector
            num_tokens (int): number of tokens allowed. Usually the vocabulary size + 2
            embedding_dim (int): embeddings dimesion. Default is 50

        Returns:
            embeddings_matrix (array), hits (int), misses (int): Returns an array 
            with all embeddings found and the number of hits and misses.
    '''
    hits = 0
    misses = 0

    # Prepare embedding matrix
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None and embedding_vector.size == embedding_dim:
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1

    return embedding_matrix, hits, misses

def create_model(embedding_layer, bidirectional=False, units=64, dropout=0.0, output_len=6):
    '''
        Creates an unidirectional LSTM model

        Parameters: 
            embedding_layer (obj): embedding layer to be used on the model
            units (int): LSTM units. Default is 64
            bidirectional (boolean): A boolean value to set the model as bidirectional
            dropout (float): dropout. Default is no dropout
            output_len (int): output length (number of the classes). Default is 6

        Returns:
            model (obj): A keras model created
    '''
    inputs = keras.Input(shape=(None,), dtype=tf.int64)

    embedded_sequences = embedding_layer(inputs)
    
    if bidirectional:
        x = layers.Bidirectional(layers.LSTM(units))(embedded_sequences)
    else:
        x = layers.LSTM(units)(embedded_sequences)

    x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(output_len, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

def run_model(model, x, y, x_val, y_val, model_name, batch_size=128, epochs=50):
    '''
        Run model with the parameters specified by the user and save the best
        weights on "model.best_weights.hdf5". The best weight is based the weights
        of the epoch with the minimum loss

        Parameters: 
            model (obj): keras model
            x (array): training data
            y (array): training labels
            x_val (array): validation data
            y_val (array): validation labels
            batch_size (int): Batch size. Default is 128
            epochs (int): Number of epochs. Default is 50

        Returns:
            history (obj): history of executed model
    '''
    checkpoint_filepath = 'results/model.best_weights-' + model_name + '.hdf5'
    model_checkpoint_callback = callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                          save_weights_only=True,
                                                          monitor='val_loss',
                                                          mode='min',
                                                          save_best_only=True)

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    history = model.fit(x, y, 
                        batch_size=batch_size, 
                        epochs=epochs,
                        verbose=0,
                        validation_data=(x_val, y_val),
                        callbacks=[model_checkpoint_callback])
    return history
