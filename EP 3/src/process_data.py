import pandas as pd
import numpy as np

DATASET_PATH = "./data/b2w-10k.csv"

MAX_TOKENS    = 7500  # numero  maximo de tokens
OUTPUT_LEN    = 45    # tamanho da saida (pad/truc)
BATCH_SIZE    = 64    # tamanho do batch
EPOCHS        = 150   # numero de epocas

def read_and_clean_data():
    # reading CSV dataset into pandas dataframe
    b2w_dataframe = pd.read_csv(DATASET_PATH)

    # selecting columns (x: 'review_text', y: 'review_title')
    processed_data = b2w_dataframe[['review_text', 'review_title']]

    # dropping NaN values
    processed_data = processed_data.dropna()

    return processed_data

def shuffle_and_split_data(dataframe):
    # shuffling data
    dataframe = dataframe.sample(frac=1).reset_index(drop=True)

    # splitting data
    validation_data_offset = int(.65 * len(dataframe))
    testing_data_offset = int(.75 * len(dataframe))

    train_df, validate_df, test_df = np.split(dataframe, [validation_data_offset, testing_data_offset])

    # reseting indexes
    train_df = train_df.reset_index()
    validate_df = validate_df.reset_index()
    test_df = test_df.reset_index()

    return train_df, validate_df, test_df

def main():
    print("Process B2W dataset (10k)")

    processed_df = read_and_clean_data()
    train_df, validate_df, test_df = shuffle_and_split_data(processed_df)

    # saving dataframe into files
    processed_df.to_csv(r'data/b2w-processed.csv', index = False, header=True)
    train_df.to_csv(r'data/b2w-train.csv', index = False, header=True)
    validate_df.to_csv(r'data/b2w-validate.csv', index = False, header=True)
    test_df.to_csv(r'data/b2w-test.csv', index = False, header=True)

    # printing daframe info
    num_samples          = len(processed_df['review_text'])
    num_train_samples    = len(train_df['review_text'])
    num_validate_samples = len(validate_df['review_text'])
    num_test_samples     = len(test_df['review_text'])

    print("Total samples: %d" % num_samples)
    print("Total samples (train córpus):      %d (%d%%)" % (num_train_samples, round(num_train_samples / num_samples * 100)))
    print("Total samples (validation córpus): %d (%d%%)" % (num_validate_samples, round(num_validate_samples / num_samples * 100)))
    print("Total samples (test córpus):       %d (%d%%)" % (num_test_samples, round(num_test_samples / num_samples * 100)))

if __name__ == "__main__":
    main()
