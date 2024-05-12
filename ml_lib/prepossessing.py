import os
import pickle
from typing import Tuple, List
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import save_to_pickle_file, load_training_params

MAX_SEQUENCE_LENGTH = 200
OOV_TOKEN = "-n-"
OUTPUT_PATH = os.path.join("data", "tokenized")

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

class Preprocessing:
    """
    Class to preprocess URL datasets
    """

    def __init__(self):
        print("Preprocessing initialized. Running preprocess...")
        self.preprocess()

    def load_dataset(self, data_path: str) -> Tuple[List[str], List[str]]:
        """Loads the data split from the path. The path should be a .txt file that
        has been created from the get_data step. This should be stored in the data folder.

        Args:
            data_path (str): The path to the split .txt file.

        Returns:
            Tuple[List[str], List[str]]: Returns a tuple of raw_x and raw_y. raw_x is a
            list of strings for all the sentences in the split and raw_y is their corresponding label.
        """
        print(f"Loading dataset from: {data_path}")
        try:
            with open(data_path, "r") as data_file:
                loaded_data = [line.strip() for line in data_file.readlines()[1:]]
        except FileNotFoundError as file_not_found_error:
            raise FileNotFoundError(f"Could not find file {data_path}.") from file_not_found_error
        except OSError as exception:
            raise OSError(f"An error occurred accessing file {data_path}: {exception}") from exception

        raw_x = [line.split("\t")[1] for line in loaded_data]
        raw_y = [line.split("\t")[0] for line in loaded_data]

        return raw_x, raw_y

    def preprocess(self, dataset_dir):
        raw_x_train, raw_y_train = self.load_dataset(os.path.join(dataset_dir, "train.txt"))
        raw_x_test, raw_y_test = self.load_dataset(os.path.join(dataset_dir, "test.txt"))
        raw_x_val, raw_y_val = self.load_dataset(os.path.join(dataset_dir, "val.txt"))

        tokenizer = Tokenizer(lower=True, char_level=True, oov_token=OOV_TOKEN)
        tokenizer.fit_on_texts(raw_x_train + raw_x_val + raw_x_test)

        encoder = LabelEncoder()

        # Save entire objects
        save_to_pickle_file(obj=tokenizer, pickle_path=os.path.join(OUTPUT_PATH, "tokenizer.pkl"))
        save_to_pickle_file(obj=encoder, pickle_path=os.path.join(OUTPUT_PATH, "label_encoder.pkl"))
class PreprocessingUtil:
    def __init__(self):
        self.tokenizer = self.load_pickle(os.path.join(OUTPUT_PATH, "tokenizer.pkl"))
        self.encoder = self.load_pickle(os.path.join(OUTPUT_PATH, "label_encoder.pkl"))

    def load_pickle(self, file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)

    def preprocess_url(self, url):
        sequences = self.tokenizer.texts_to_sequences([url])
        padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        return padded_sequences
