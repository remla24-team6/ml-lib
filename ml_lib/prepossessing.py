import os
import pickle
from typing import Tuple, List, Any
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from importlib_resources import files


class PreprocessingUtil:
    """
    Fits the label encoder and tokenizer on the entire phishing detection dataset and saves them locally.

    In order to perform the pre-processing, we need the tokenizer and the label encoder that is fit on the entire dataset. One possibility is to
    fit the tokenizer and the encoder on the entire data in the package itself (as part of the init of the Preprocessing class below). However, that
    would bind the data with the library. Instead, we use this utility class to fit the tokenizer and label encoder and save them as pickle files.
    This would allow us to push them as assets as part of the package itself instead of binding the data to the library.
    """

    def __init__(self, save_path: str, dataset_dir: str, oov_token: str = "-n-"):
        """_summary_

        Args:
            save_path (str): Path where the tokenizer and label encoder pickle files will be saved.
            dataset_dir (str): Path where the dataset splits are saved. This directory should have train.txt, test.txt, val.txt saved.
            oov_token (str): The string for a token that is out of the tokenizer's vocabulary. Defaults to "-n-".
        """
        self.save_path = save_path
        self.dataset_dir = dataset_dir
        self.oov_token = oov_token

    @staticmethod
    def _save_to_pickle_file(obj: Any, pickle_path: str) -> Any:
        """Saves the given object into a pickle file defined by the pickle_path

        Args:
            obj: (Any): The object that needs to be dumped.
            pickle_path (str): Path to pickle file where object will be stored.

        Returns:
            Any: Returns the loaded object.
        """

        with open(pickle_path, "wb") as f:
            pickle.dump(obj, f)

    @staticmethod
    def _load_dataset(data_path: str) -> Tuple[List[str], List[str]]:
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
            raise FileNotFoundError(
                f"Could not find file {data_path}."
            ) from file_not_found_error
        except OSError as exception:
            raise OSError(
                f"An error occurred accessing file {data_path}: {exception}"
            ) from exception

        raw_x = [line.split("\t")[1] for line in loaded_data]
        raw_y = [line.split("\t")[0] for line in loaded_data]

        return raw_x, raw_y

    def fit_and_save_assets(self):
        """Fits the tokenizer and the label encoder on the datasets and saves them."""
        raw_x_train, raw_y_train = self._load_dataset(
            os.path.join(self.dataset_dir, "train.txt")
        )
        raw_x_test, _ = self._load_dataset(os.path.join(self.dataset_dir, "test.txt"))
        raw_x_val, _ = self._load_dataset(os.path.join(self.dataset_dir, "val.txt"))

        tokenizer = Tokenizer(lower=True, char_level=True, oov_token=self.oov_token)
        tokenizer.fit_on_texts(raw_x_train + raw_x_val + raw_x_test)

        encoder = LabelEncoder()
        encoder.fit(raw_y_train)

        self._save_to_pickle_file(
            obj=encoder, pickle_path=os.path.join(self.save_path, "label_encoder.pkl")
        )
        self._save_to_pickle_file(
            obj=tokenizer, pickle_path=os.path.join(self.save_path, "tokenizer.pkl")
        )


class Preprocessing:
    """Performs the pre-processing for phishing detection."""

    def __init__(self, max_sequence_length: int = 200):
        """
        max_sequence_length (optional): The maximum sequence length for the tokenizer. Defaults to 200.
        """
        self.max_sequence_length = max_sequence_length
        self.tokenizer = self._load_pickle("tokenizer.pkl")
        self.encoder = self._load_pickle("label_encoder.pkl")

    @staticmethod
    def _load_pickle(file_name: str):
        """Loads the pickle file in a package-friendly manner. This pickle file should be in the resources directory and
        should be included in the pyproject.toml file so that it can be used within the package.

        Args:
            file_name (str): Name of the file to load.
        """

        file_path = files('resources').joinpath(file_name)
        with open(file_path, 'rb') as file:
            loaded_file = pickle.load(file)
            
        return loaded_file

    def tokenize_batch(self, urls: List[str]):
        """Performs the tokenization on a list of urls. This can be used to pre-process the entire dataset.

        Args:
            urls (List[str]): The list of urls to be tokenized.

        Returns:
            np.ndarray: Returns a numpy array with shape (len(urls), maxlen).
        """
        return pad_sequences(
            self.tokenizer.texts_to_sequences(urls), maxlen=self.max_sequence_length
        )
    
    def tokenize_single(self, url: str):
        """Performs the tokenization for a single url. This can be used to pre-process a single url at inference time.

        Args:
            url (str): The url to be tokenized.

        Returns:
            np.ndarray: Returns a numpy array with shape (maxlen).
        """
        return pad_sequences(
            self.tokenizer.texts_to_sequences([url]), maxlen=self.max_sequence_length
        )[0]
    
    def encode_label_batch(self, labels: List[str]):
        """Performs the label encoding for a list of labels. This can be used to pre-process the target labels of the entire dataset.

        Args:
            labels (List[str]): List of labels that need to be encoded.

        Returns:
            An arrat of shape (len(labels)).
        """
        return self.encoder.transform(labels)