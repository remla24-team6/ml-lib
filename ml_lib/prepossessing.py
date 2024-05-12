from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gdown

class Preprocessing:
    """
    Class to preprocess URL datasets
    """
    def __init__(self):

        gdown.download_folder('https://drive.google.com/drive/folders/1_NobSEMZS8jogSAEZ9ZLBUTemPiASJRg',
                              output="data", quiet=False)

        train = [line.strip() for line in open('data/train.txt', "r", encoding="utf-8").readlines()[1:]]
        x_train = [line.split("\t")[1] for line in train]
        y_train = [line.split("\t")[0] for line in train]
        test = [line.strip() for line in open('data/test.txt', "r", encoding="utf-8").readlines()]
        x_test = [line.split("\t")[1] for line in test]
        val = [line.strip() for line in open('data/val.txt', "r", encoding="utf-8").readlines()]
        x_val = [line.split("\t")[1] for line in val]

        self.tokenizer = Tokenizer(lower=True, char_level=True, oov_token='-n-')  # nosec
        self.tokenizer.fit_on_texts(x_train + x_val + x_test)
        self.sequence_length = 200
        self.encoder = LabelEncoder()
        self.encoder = self.encoder.fit(y_train)
    def process_dataset(self, dataset):
        return pad_sequences(self.tokenizer.texts_to_sequences(dataset), maxlen=self.sequence_length)
    def process_URL(self, url):
        return pad_sequences(self.tokenizer.texts_to_sequences([url]), maxlen=self.sequence_length)[0]
    def process_labels(self, labels):
        return self.encoder.transform(labels)
    def process_label(self, label):
        return self.encoder.transform([label])[0]
    def get_char_index(self):
        return self.tokenizer.word_index