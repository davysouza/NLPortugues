import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random

import tensorflow as tf

from transformers import (
    BertTokenizer,
    TFBertForSequenceClassification,
    TFTrainer,
    TFTrainingArguments,
)

from sklearn.model_selection import train_test_split

# max of words per review text
MAX_INPUT_WORD = 30

# number of wordpiece tokens created by bert tokenizer
MAX_TOKENS_LEN = 100

EPOCHS = 5
BATCH_SIZE = 16

# dataset paths
DATASET_PATH_ALL      = "./data/b2w-processed.csv"
DATASET_PATH_TRAIN    = "./data/b2w-train.csv"
DATASET_PATH_VALIDATE = "./data/b2w-validate.csv"
DATASET_PATH_TEST     = "./data/b2w-test.csv"


# creating vocab based on data
# return: set of words + map word to index
def create_vocab(data):    
    y_vocab_set = set()
    for title in data:
        splitted_title = title.split()
        for word in splitted_title:
            y_vocab_set.add(word.lower())
    y_vocab_set.add('[SEP]')
    y_vocab_set.add('[UNK]')
    y_vocab_set.add('<EOS>')
    
    y_vocab_list = sorted(list(y_vocab_set))
    word_to_idx  = dict([(char, i) for i, char in enumerate(y_vocab_list)])
    
    return y_vocab_list, word_to_idx

# process data adding tokens [SEP], [MASK]
# return: processed data on format: review + [SEP] + title + [MASK] + [SEP]
def processing_data(data, max_words_input=60):
    reviews = data[['review_text']].values.tolist()
    titles  = np.array([s for [s] in data[['review_title']].values.tolist()])
    
    x = []
    y = []
    for review, title in zip(reviews, titles):
        splitted_review = review[0].split()
        splitted_title  = title.split()
        splitted_review = splitted_review[:min(max_words_input - len(splitted_title) - 1, len(splitted_review))]

        partial_title = ''
        for word in splitted_title:
            x.append([' '.join(splitted_review) + ' [SEP]' + partial_title + ' [MASK]'])
            partial_title += ' ' + word
            y.append(word.lower())

        x.append([' '.join(splitted_review) + ' [SEP] ' + title])
        y.append('<EOS>')
        
    return x, y

# create an embedding array of words
def get_embedded_tokens(words, word_to_idx):
    emb_tokens = []
    for word in words:
        emb_tokens.append(word_to_idx[word])
    
    return np.array(emb_tokens)

# plot accuracy and loss throught epochs
def plot_results(history):
    # Plot graphic of accuracy
    plt.cla()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('BERT model (Accuracy)')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('images/bert_model_accuracy.png')

    # Plot graphic of loss
    plt.cla()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('BERT model (Loss)')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('images/bert_model_loss.png')

# tokenize data using bert tokenizer
def tokenize_data(tokenizer, word_to_index, x, y):
    encoded_x = tokenizer(
        x,
        text_pair=None,
        is_split_into_words=True,
        padding="max_length",
        truncation=True,
        max_length=MAX_TOKENS_LEN,
        pad_to_max_length=True,
        return_tensors='tf'
    )

    # cria one-hot para dados de treinamento
    encoded_y = get_embedded_tokens(y, word_to_index)

    return encoded_x, encoded_y

def main():
    # load dataframe
    dataframe   = pd.read_csv(DATASET_PATH_ALL)
    train_df    = pd.read_csv(DATASET_PATH_TRAIN)
    validate_df = pd.read_csv(DATASET_PATH_VALIDATE)
    test_df     = pd.read_csv(DATASET_PATH_TEST)

    # create vocabulary and word_to_index from all words of titles
    all_titles = np.array([s for [s] in dataframe[['review_title']].values.tolist()])
    y_vocab, word_to_index = create_vocab(all_titles)

    # processing data
    x_train, y_train = processing_data(train_df, MAX_INPUT_WORD)
    x_validate, y_validate = processing_data(validate_df, MAX_INPUT_WORD)
    x_test, y_test = processing_data(test_df, MAX_INPUT_WORD)

    # BERT tokenizer
    REF_MODEL = 'neuralmind/bert-base-portuguese-cased'
    tokenizer = BertTokenizer.from_pretrained(REF_MODEL)

    # encoding data
    encoded_train, encoded_train_labels = tokenize_data(tokenizer, word_to_index, x_train, y_train)
    encoded_validate, encoded_validate_labels = tokenize_data(tokenizer, word_to_index, x_validate, y_validate)
    encoded_test, encoded_test_labels = tokenize_data(tokenizer, word_to_index, x_test, y_test)

    # output len is the size of the vocabulary
    output_len = len(word_to_index)

    # 
    # create model
    # 

    # BERT encoder
    bert_model = TFBertForSequenceClassification.from_pretrained(REF_MODEL, from_pt=True, num_labels=output_len)

    # input layers
    input_ids = tf.keras.layers.Input(shape=(MAX_TOKENS_LEN,), name='input_token', dtype='int32')
    input_masks_ids = tf.keras.layers.Input(shape=(MAX_TOKENS_LEN,), name='masked_token', dtype='int32')

    # decode
    X = bert_model(input_ids, input_masks_ids)[0]
    X = tf.keras.layers.Dense(128, activation='relu')(X)
    X = tf.keras.layers.Dropout(.4)(X)
    X = tf.keras.layers.Dense(output_len, activation='sigmoid')(X)

    model = tf.keras.Model(inputs=[input_ids, input_masks_ids], outputs=X)

    # compiling model
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=['accuracy'])

    # callbacks
    checkpoint_filepath = 'history/bert_model.best.hdf5'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                                   save_weights_only=True,
                                                                   monitor='val_loss',
                                                                   mode='min',
                                                                   save_best_only=True)


    
    training
    
    history = model.fit([encoded_train["input_ids"], encoded_train["attention_mask"]],
                        encoded_train_labels,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=([encoded_validate["input_ids"], encoded_validate["attention_mask"]], encoded_validate_labels),
                        callbacks=[model_checkpoint_callback])
    
    # plotting accuracy and loss
    plot_results(history)

    # 
    # testing
    # 
    model.load_weights('history/bert_model.best.hdf5')

    # evaluate
    model.evaluate(x=[encoded_test["input_ids"], encoded_test["attention_mask"]], y=np.array(encoded_test_labels))

    # predicting 10 samples
    for u in range(10):
        rand_idx = random.randint(1, 2000)
        print("Sample: %d\n" % (u + rand_idx))
        
        # adding tokens
        sample = '[CLS] ' + x_test[u+rand_idx][0] + ' [SEP]'
        print("Sample: %s\n" % sample)

        # tokenizing review
        tokenized_sample = tokenizer.tokenize(sample)
        print("Tokenized sample: %s\n" % tokenized_sample)
        
        # tokens to ids
        idx_sample = tokenizer.convert_tokens_to_ids(tokenized_sample)
        print("Tokenized ids sample: %s\n" % idx_sample)
        
        # creating numpy array of ids
        sample_token_ids = np.zeros(MAX_TOKENS_LEN, dtype=int)
        for i in range(len(idx_sample)):
            sample_token_ids[i] = idx_sample[i]

        # creating mask
        mask_input = np.zeros(MAX_TOKENS_LEN, dtype=int)
        for i in range(len(idx_sample)):
            mask_input[i] = 1

        # predicting
        prediction = model.predict([sample_token_ids, mask_input])

        # get biggest value
        word_idx = prediction.argmax(axis=1)[0]

        # index to word
        word = y_vocab[word_idx]
                        
        print('Predição: %s' % word)
        print('Esperado: %s\n'% y_test[u+rand_idx])
        print('============================================================\n')

if __name__ == "__main__":
    main()
