import pandas as pd
import numpy as np

import nltk
import nltk.translate as translate
import nltk.translate.nist_score as nist_score

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model

# dataset paths
DATASET_PATH_ALL      = "./data/b2w-processed.csv"
DATASET_PATH_TRAIN    = "./data/b2w-train.csv"
DATASET_PATH_VALIDATE = "./data/b2w-validate.csv"
DATASET_PATH_TEST     = "./data/b2w-test.csv"

MAX_TOKENS    = 30000 # max number of tokens
OUTPUT_LEN    = 45    # output length (pad/truc)
BATCH_SIZE    = 16    # batch size
EPOCHS        = 100   # number of epochs

Tx = 45 # size of x
Ty =  6 # size of y

n_a = 32 # number of LSTM units
n_s = 64 # size of input layer

# 
# model
# 
repeat_layer = tf.keras.layers.RepeatVector(Tx)
concat_layer = tf.keras.layers.Concatenate()
dense_layer  = tf.keras.layers.Dense(32)
activ_layer  = tf.keras.layers.Dense(32, activation="softmax")
dot_layer    = tf.keras.layers.Dot(axes=1)

def one_step_attention(h, s_prev):
    """
    Defina a função de atenção e retorne o vetor contexto 
    """
    s_prev     = repeat_layer(s_prev)
    concatted  = concat_layer([h, s_prev])
    dense      = dense_layer(concatted)
    activation = activ_layer(dense)
    context    = dot_layer([activation, h])
    
    return context

def create_model(vocab_size):
    post_activation_LSTM_cell = tf.keras.layers.LSTM(n_s, return_state=True, dropout=.25, recurrent_dropout=.25) 
    output_layer = tf.keras.layers.Dense(vocab_size, activation="softmax")

    X  = tf.keras.Input(shape=(Tx, vocab_size))
    s0 = tf.keras.Input(shape=(n_s,), name='s0')
    c0 = tf.keras.Input(shape=(n_s,), name='c0')
    s = s0
    c = c0
    
    outputs = []
    
    # Encoder (Bi-LSTM)
    h = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_a, return_sequences=True, dropout=.25, recurrent_dropout=.25))(X)
    
    for t in range(Ty):
        # Decoder (Attention + LSTM)
        context = one_step_attention(h, s)
        s, _, c = post_activation_LSTM_cell(context, initial_state=[s, c])
        out = output_layer(s)
        outputs.append(out)
    
    model = tf.keras.Model(inputs=[X, s0, c0], outputs=outputs)
    
    return model

# 
# utils
# 

def string_to_int(string, length, vocab):
    string = string.lower()
    string = string.replace(',','')
    
    arr = string.split()
        
    X = []
    idx = 0
    while idx < length:
        if idx >= len(arr):
            X.append(vocab[''])
        else:
            if arr[idx] in vocab:
                X.append(vocab[arr[idx]])
            else:
                X.append(vocab['[UNK]'])
                
        idx += 1        
    return X

# create one hot encoding from dataset based on vocab
def create_onehot(dataset, vocab):
    X, Y = zip(*dataset)
        
    X = np.array([string_to_int(i, Tx, vocab) for i in X])
    Y = [string_to_int(t, Ty, vocab) for t in Y]
    
    # create an one-hot encoding of X and Y based on vocabulary
    Xoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(vocab)), X)))
    Yoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(vocab)), Y)))

    return X, np.array(Y), Xoh, Yoh

# process dataframe
# returns: an array with a pair (review, title)
def process_dataframe(dataframe):
    dataset = []
    for i in range(len(dataframe)):
        dataset.append((dataframe['review_text'][i], dataframe['review_title'][i]))
    return dataset

# plot accuracy and loss throught epochs
def plot_results(history):
    # Plot graphic of accuracy
    plt.cla()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Bi-LSTM with Attention model (Accuracy)')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('images/bi-lstm_model_accuracy.png')

    # Plot graphic of loss
    plt.cla()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Bi-LSTM with Attention model (Loss)')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('images/bi-lstm_model_loss.png')

def main():
    # loading dataframe
    dataframe   = pd.read_csv(DATASET_PATH_ALL)
    train_df    = pd.read_csv(DATASET_PATH_TRAIN)
    validate_df = pd.read_csv(DATASET_PATH_VALIDATE)
    test_df     = pd.read_csv(DATASET_PATH_TEST)

    # adding startsent and endsent tokens
    dataframe['review_title']   = ['startsent ' + dataframe['review_title'][i] + ' endsent' for i in range(len(dataframe['review_title']))]
    train_df['review_title']    = ['startsent ' + train_df['review_title'][i] + ' endsent' for i in range(len(train_df['review_title']))]
    validate_df['review_title'] = ['startsent ' + validate_df['review_title'][i] + ' endsent' for i in range(len(validate_df['review_title']))]
    test_df['review_title']     = ['startsent ' + test_df['review_title'][i] + ' endsent' for i in range(len(test_df['review_title']))]

    # vectorzing data
    vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=MAX_TOKENS, output_sequence_length=OUTPUT_LEN)
    text_ds = tf.data.Dataset.from_tensor_slices(dataframe['review_text'] + dataframe['review_title']).batch(BATCH_SIZE)
    vectorizer.adapt(text_ds)

    # creating vocabulary
    vocabulary = vectorizer.get_vocabulary()
    vocabulary_size = len(vocabulary)
    print("vocabulary size: %d" % vocabulary_size)

    # vocabulary and inverted vocabulary
    word_index = dict(zip(vocabulary, range(vocabulary_size)))
    index_word = dict(enumerate(vocabulary))

    # create an array of a pair (review, title)
    dataset_train  = process_dataframe(train_df)
    dataset_val    = process_dataframe(validate_df)
    dataset_test   = process_dataframe(test_df)

    # create an array of integers and their respectives one-hot encodings
    X, Y, Xoh, Yoh = create_onehot(dataset_train, word_index)
    X_val, Y_val, Xoh_val, Yoh_val = create_onehot(dataset_val, word_index)
    X_test, Y_test, Xoh_test, Yoh_test = create_onehot(dataset_test, word_index)

    model = create_model(len(word_index))
    model.compile(optimizer="rmsprop", metrics=['accuracy'], loss='categorical_crossentropy')

    # 
    # Training
    # 
    num_train_samples    = len(train_df['review_text'])
    num_validate_samples = len(validate_df['review_text'])
    num_test_samples     = len(test_df['review_text'])

    s0 = np.zeros((num_train_samples, n_s))
    c0 = np.zeros((num_train_samples, n_s))
    outputs = list(Yoh.swapaxes(0, 1))

    s0_val = np.zeros((num_validate_samples, n_s))
    c0_val = np.zeros((num_validate_samples, n_s))
    outputs_val = list(Yoh_val.swapaxes(0, 1))

    # callbacks
    checkpoint_filepath = 'history/bi-lstm_model.best.hdf5'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                                   save_weights_only=True,
                                                                   monitor='val_loss',
                                                                   mode='min',
                                                                   save_best_only=True)

    history = model.fit([Xoh, s0, c0], 
                        outputs, 
                        epochs=EPOCHS, 
                        batch_size=BATCH_SIZE, 
                        validation_data=([Xoh_val, s0_val, c0_val], outputs_val),
                        callbacks=[model_checkpoint_callback])

    # load best model
    model.load_weights('history/bi-lstm_model.best.hdf5')

    # evaluate
    s0_test = np.zeros((num_test_samples, n_s))
    c0_test = np.zeros((num_test_samples, n_s))
    outputs_test = list(Yoh_test.swapaxes(0, 1))
    model.evaluate(x=[Xoh_test, s0_test, c0_test], y=outputs_test)

    # 
    # Testing
    # 
    test_dataset = process_dataframe(test_df)
    test_dataset = [test_dataset[i][0] for i in range(len(test_dataset))]

    # Download do wordnet
    nltk.download('wordnet')

    with open('output_metrics_bi-lstm.txt', 'w', encoding='utf-16-le') as file:
        # for each sample in test dataset
        for example, df in zip(test_dataset, test_df['review_title']):
            # samples to one-hot encoding
            source = string_to_int(example, Tx, word_index)
            source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(word_index)), source)))

            source = source.reshape((1, Tx, len(word_index)))

            s0 = np.zeros((1, n_s))
            c0 = np.zeros((1, n_s))

            # predicting title
            prediction = model.predict([source, s0, c0])
            prediction = np.argmax(prediction, axis=-1)

            output = [index_word[int(i)] for i in prediction]

            # processing results (removing starsent and endsent tokens)
            processed_out = ''
            for w in output:
                if w != 'startsent' and w != 'endsent' and w != '':
                    processed_out += w + ' '

                if w ==  'endsent':
                    break

            expected = []
            for i in df.split():
                if i != 'endsent' and i != 'startsent':
                    expected.append(i)

            # saving results
            file.write("input:    %s\n" % example)
            file.write("output:   %s\n" % processed_out)
            file.write("expected: %s\n\n" % ' '.join(expected))

            # metrics
            file.write("BLEU:   %f\n" % translate.bleu_score.sentence_bleu([expected], processed_out.split()))
            file.write("METEOR: %f\n" % translate.meteor_score.meteor_score([' '.join(expected)], processed_out))
            file.write("NIST:   %f\n\n" % nist_score.sentence_nist([expected], processed_out.split(), 1))

            file.write("==================================================================\n\n\n")

if __name__ == "__main__":
    main()
