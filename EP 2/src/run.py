import process_data as process_data
import model

import pandas as pd
import numpy as np

import time
import json

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow import data

from tensorflow import keras
from tensorflow.keras import layers

##########################################################################
#                            Definitions                                 #
##########################################################################

# some datasets that can be used
b2w_10k_dataset_path  = "data/b2w-10k.csv"
b2w_full_dataset_path = "data/B2W-Reviews01.csv"

# some word embeddings
cbow_s50_path   = "data/cbow_s50.txt"
cbow_s100_path  = "data/cbow_s100.txt"
cbow_s300_path  = "data/cbow_s300.txt"
cbow_s600_path  = "data/cbow_s600.txt"
cbow_s1000_path = "data/cbow_s1000.txt"

# cbow_s50 - 200k embeddings 
word2vec_200k_path = "data/word2vec_200k.txt"

# classes
class_names = ["Unknown", "Pessimo", "Ruim", "OK", "Bom", "Otimo"]

# required columns
selected_columns = ['review_text', 'overall_rating']


##########################################################################
#                             Variables                                  #
##########################################################################

DATASET_PATH = b2w_10k_dataset_path
WORD_EMBEDDING_PATH = cbow_s50_path

MAX_TOKENS = 20000
OUTPUT_LEN = 50
BATCH_SIZE = 128
EMBEDDING_DIM = 50

##########################################################################


def test_vectorization(vectorizer, word_index):
    print('\n==> Vectorizer Test')

    vocabulary = vectorizer.get_vocabulary()
    vocabulary_size = len(vocabulary)

    print('   Vocabulary size: %d' % vocabulary_size)
    print('   Vectorizer head: %s\n' % vocabulary[:5])

    test_str = "era exatamente o que eu queria"
    test_arr = ["era", "exatamente", "o", "que", "eu", "queria"]

    output = vectorizer([[test_str]])
    print('   Test string: %s' % test_str)
    print('     vectorizer: %s' % output.numpy()[0, :6])
    print('     word_index: %s' % [word_index[w] for w in test_arr])

def plot_results(history, model_name):
    # Plot graphic of accuracy
    plt.cla()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('images/' + model_name + '_accuracy.png')

    # Plot graphic of loss
    plt.cla()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('images/' + model_name + '_loss.png')

def train_model(keras_model, x, y, x_val, y_val, model_name, batch_size=128, epochs=50):
    history = model.run_model(keras_model, x, y, x_val, y_val, model_name=model_name, batch_size=batch_size, epochs=epochs)
    plot_results(history, model_name)

    # print results
    best_loss  = min(history.history['val_loss'])
    best_epoch = history.history['val_loss'].index(best_loss)
    print('  - Best results:')
    print('    - Epoch: %d' % best_epoch)
    print('    -  Loss: %.6f' % best_loss)

    # save history
    with open('history/history_' + model_name + '.json', "w", encoding="utf8") as file:
        file.write(json.dumps(history.history, indent=4))


def main():
    '''
        Main function
    '''

    ##########################################################################
    #                      Pre-preocessing dataset                           #
    ##########################################################################

    print('\n\n\n========== Starting dataset pre-processing ===========\n')

    print('Reading CSV file...\n')
    if DATASET_PATH == b2w_full_dataset_path:
        b2w_dataframe = pd.read_csv(DATASET_PATH, sep=";")
    else:
        b2w_dataframe = pd.read_csv(DATASET_PATH)

    print('Cleaning data...\n')
    b2w_dataframe_processed = process_data.clean_data(b2w_dataframe, selected_columns)

    print("Found %5d samples of class 1" % len(b2w_dataframe_processed[b2w_dataframe_processed['overall_rating'] == 1]))
    print("Found %5d samples of class 2" % len(b2w_dataframe_processed[b2w_dataframe_processed['overall_rating'] == 2]))
    print("Found %5d samples of class 3" % len(b2w_dataframe_processed[b2w_dataframe_processed['overall_rating'] == 3]))
    print("Found %5d samples of class 4" % len(b2w_dataframe_processed[b2w_dataframe_processed['overall_rating'] == 4]))
    print("Found %5d samples of class 5" % len(b2w_dataframe_processed[b2w_dataframe_processed['overall_rating'] == 5]))

    print('\nDataset sample:\n%s\n' % b2w_dataframe_processed.head())

    print('Shuffling and splitting data...\n')
    train_df, validate_df, test_df = process_data.shuffle_and_split(b2w_dataframe_processed)

    num_samples          = len(b2w_dataframe_processed['review_text'])
    num_train_samples    = len(train_df['review_text'])
    num_validate_samples = len(validate_df['review_text'])
    num_test_samples     = len(test_df['review_text'])

    print("Total of samples found: %d" % num_samples)
    print("Total of training samples found: %d (%d%%)" % (num_train_samples, round(num_train_samples / num_samples * 100)))
    print("Total of validation samples found : %d (%d%%)" % (num_validate_samples, round(num_validate_samples / num_samples * 100)))
    print("Total of testing samples found: %d (%d%%)\n" % (num_test_samples, round(num_test_samples / num_samples * 100)))

    print('Saving data...\n')
    train_df.to_csv(r'b2w-train.csv', index = False, header=True)
    validate_df.to_csv(r'b2w-validate.csv', index = False, header=True)
    test_df.to_csv(r'b2w-test.csv', index = False, header=True)

    print('Vectorizing data...\n')
    vectorizer = process_data.create_vectorizer(train_df['review_text'], MAX_TOKENS, OUTPUT_LEN, BATCH_SIZE)

    # Vocabulary
    vocabulary = vectorizer.get_vocabulary()
    vocabulary_size = len(vocabulary)

    print('Vocabulary size: %d' % vocabulary_size)

    # Create a word index
    word_index = dict(zip(vocabulary, range(vocabulary_size)))
    test_vectorization(vectorizer, word_index)

    # Preparing data to be used in the model
    x_train_embedded, y_train_embedded = process_data.vectorize_data(vectorizer, train_df['review_text'], train_df['overall_rating'])
    x_validate_embedded, y_validate_embedded = process_data.vectorize_data(vectorizer, validate_df['review_text'], validate_df['overall_rating'])
    x_test_embedded, y_test_embedded = process_data.vectorize_data(vectorizer, test_df['review_text'], test_df['overall_rating'])



    ##########################################################################
    #                               Learning                                 #
    ##########################################################################

    # Load pre-trained word embeddings
    embeddings_index = model.load_word_embedding(WORD_EMBEDDING_PATH)
    print("Found %s word vectors." % len(embeddings_index))

    # Create a embedding matrix from word_index using embeddings index
    num_tokens = vocabulary_size + 2
    embedding_matrix, hits, misses = model.create_embedding_matrix(word_index, embeddings_index, num_tokens, EMBEDDING_DIM)
    print("Converted %d words (%d misses)" % (hits, misses))

    # Creating an embedding layer
    embedding_layer = layers.Embedding(num_tokens,
                                       EMBEDDING_DIM,
                                       embeddings_initializer=keras.initializers.Constant(embedding_matrix),
                                       trainable=False)

    
    
    # ====================== Running and evaluating ==========================


    # LSTM model without dropout
    print('\n\nRunning unidirectional LSTM model...\n')
    lstm_model = model.create_model(embedding_layer)
    train_model(lstm_model, 
                x_train_embedded, 
                y_train_embedded, 
                x_validate_embedded, 
                y_validate_embedded, 
                'unidirect_lstm', 
                batch_size=BATCH_SIZE, 
                epochs=50)

    # Load and evaluate results
    lstm_model.load_weights('results/model.best_weights-unidirect_lstm.hdf5')
    lstm_model.evaluate(x=x_test_embedded, y=y_test_embedded)

    
    # ========================================================================


    # LSTM model (dropout = 25%)
    print('\n\nRunning unidirectional LSTM model (dropout: .25)...\n')
    lstm_model = model.create_model(embedding_layer, dropout=.25)
    train_model(lstm_model, 
                x_train_embedded,
                y_train_embedded,
                x_validate_embedded,
                y_validate_embedded, 
                'unidirect_lstm_drop25',
                batch_size=BATCH_SIZE,
                epochs=50)

    # Load and evaluate results
    lstm_model.load_weights('results/model.best_weights-unidirect_lstm_drop25.hdf5')
    lstm_model.evaluate(x=x_test_embedded, y=y_test_embedded)


    # ========================================================================


    # LSTM model (dropout = 50%)
    print('\n\nRunning unidirectional LSTM model (dropout: .50)...\n')
    lstm_model = model.create_model(embedding_layer, dropout=.50)
    train_model(lstm_model, 
                x_train_embedded,
                y_train_embedded,
                x_validate_embedded,
                y_validate_embedded, 
                'unidirect_lstm_drop50',
                batch_size=BATCH_SIZE,
                epochs=50)

    # Load and evaluate results
    lstm_model.load_weights('results/model.best_weights-unidirect_lstm_drop50.hdf5')
    lstm_model.evaluate(x=x_test_embedded, y=y_test_embedded)


    # ========================================================================


    # Bidirectional LSTM model without dropout
    print('\n\nRunning bidirectional LSTM model...\n')
    bidirect_model = model.create_model(embedding_layer, bidirectional=True)
    train_model(bidirect_model, 
                x_train_embedded, 
                y_train_embedded, 
                x_validate_embedded, 
                y_validate_embedded, 
                'bidirect_lstm',
                batch_size=BATCH_SIZE,
                epochs=50)

    # Load and evaluate results
    bidirect_model.load_weights('results/model.best_weights-bidirect_lstm.hdf5')
    bidirect_model.evaluate(x=x_test_embedded, y=y_test_embedded)


    # ========================================================================


    # Bidirectional LSTM model (dropout = 25%)
    print('\n\nRunning bidirectional LSTM model (dropout: .25)...\n')
    bidirect_model = model.create_model(embedding_layer, bidirectional=True, dropout=.25)
    train_model(bidirect_model, 
                x_train_embedded, 
                y_train_embedded, 
                x_validate_embedded, 
                y_validate_embedded, 
                'bidirect_lstm_drop25',
                batch_size=BATCH_SIZE,
                epochs=50)

    # Load and evaluate results
    bidirect_model.load_weights('results/model.best_weights-bidirect_lstm_drop25.hdf5')
    bidirect_model.evaluate(x=x_test_embedded, y=y_test_embedded)


    # ========================================================================


    # Bidirectional LSTM model (dropout = 50%)
    print('\n\nRunning bidirectional LSTM model (dropout: .50)...\n')
    bidirect_model = model.create_model(embedding_layer, bidirectional=True, dropout=.50)
    train_model(bidirect_model, 
                x_train_embedded, 
                y_train_embedded, 
                x_validate_embedded, 
                y_validate_embedded, 
                'bidirect_lstm_drop50',
                batch_size=BATCH_SIZE,
                epochs=50)

    # Load and evaluate results
    bidirect_model.load_weights('results/model.best_weights-bidirect_lstm_drop50.hdf5')
    bidirect_model.evaluate(x=x_test_embedded, y=y_test_embedded)


    # ========================================================================




    # ============================ 100 epochs ================================

    print('\n\nRunning models with 100 epochs...\n')
    time.sleep(.500)

    # LSTM model without dropout 
    print('\n\nRunning unidirectional LSTM model (epochs: 100)...\n')
    lstm_model = model.create_model(embedding_layer)
    train_model(lstm_model, 
                x_train_embedded,
                y_train_embedded,
                x_validate_embedded,
                y_validate_embedded,
                'unidirect_lstm_e100',
                batch_size=BATCH_SIZE,
                epochs=100)

    # Load and evaluate results
    lstm_model.load_weights('results/model.best_weights-unidirect_lstm_e100.hdf5')
    lstm_model.evaluate(x=x_test_embedded, y=y_test_embedded)

    
    # ========================================================================


    # LSTM model (dropout = 25%)
    print('\n\nRunning unidirectional LSTM model (dropout: .25, epochs: 100)...\n')
    lstm_model = model.create_model(embedding_layer, dropout=.25)
    train_model(lstm_model, 
                x_train_embedded,
                y_train_embedded,
                x_validate_embedded,
                y_validate_embedded,
                'unidirect_lstm_drop25_e100',
                batch_size=BATCH_SIZE,
                epochs=100)

    # Load and evaluate results
    lstm_model.load_weights('results/model.best_weights-unidirect_lstm_drop25_e100.hdf5')
    lstm_model.evaluate(x=x_test_embedded, y=y_test_embedded)


    # ========================================================================


    # LSTM model (dropout = 50%)
    print('\n\nRunning unidirectional LSTM model (dropout: .50, epochs: 100)...\n')
    lstm_model = model.create_model(embedding_layer, dropout=.50)
    train_model(lstm_model, 
                x_train_embedded,
                y_train_embedded,
                x_validate_embedded,
                y_validate_embedded,
                'unidirect_lstm_drop50_e100',
                batch_size=BATCH_SIZE,
                epochs=100)

    # Load and evaluate results
    lstm_model.load_weights('results/model.best_weights-unidirect_lstm_drop50_e100.hdf5')
    lstm_model.evaluate(x=x_test_embedded, y=y_test_embedded)


    # ========================================================================


    # Bidirectional LSTM model without dropout
    print('\n\nRunning bidirectional LSTM model (epochs: 100)...\n')
    bidirect_model = model.create_model(embedding_layer, bidirectional=True)
    train_model(bidirect_model, 
                x_train_embedded,
                y_train_embedded,
                x_validate_embedded,
                y_validate_embedded,
                'bidirect_lstm_e100',
                batch_size=BATCH_SIZE,
                epochs=100)

    # Load and evaluate results
    bidirect_model.load_weights('results/model.best_weights-bidirect_lstm_e100.hdf5')
    bidirect_model.evaluate(x=x_test_embedded, y=y_test_embedded)


    # ========================================================================


    # Bidirectional LSTM model (dropout = 25%)
    print('\n\nRunning bidirectional LSTM model (dropout: .25, epochs: 100)...\n')
    bidirect_model = model.create_model(embedding_layer, bidirectional=True, dropout=.25)
    train_model(bidirect_model, 
                x_train_embedded,
                y_train_embedded,
                x_validate_embedded,
                y_validate_embedded,
                'bidirect_lstm_drop25_e100',
                batch_size=BATCH_SIZE,
                epochs=100)

    # Load and evaluate results
    bidirect_model.load_weights('results/model.best_weights-bidirect_lstm_drop25_e100.hdf5')
    bidirect_model.evaluate(x=x_test_embedded, y=y_test_embedded)


    # ========================================================================


    # Bidirectional LSTM model (dropout = 50%)
    print('\n\nRunning bidirectional LSTM model (dropout: .50, epochs: 100)...\n')
    bidirect_model = model.create_model(embedding_layer, bidirectional=True, dropout=.50)
    train_model(bidirect_model, 
                x_train_embedded,
                y_train_embedded,
                x_validate_embedded,
                y_validate_embedded,
                'bidirect_lstm_drop50_e100',
                batch_size=BATCH_SIZE,
                epochs=100)

    # Load and evaluate results
    bidirect_model.load_weights('results/model.best_weights-bidirect_lstm_drop50_e100.hdf5')
    bidirect_model.evaluate(x=x_test_embedded, y=y_test_embedded)

if __name__ == "__main__":
    main()
