
class PreprocessingUtil:
    def __init__(self):
        self.tokenizer = self.load_pickle(os.path.join(OUTPUT_PATH, "tokenizer.pkl"))
        self.encoder = self.load_pickle(os.path.join(OUTPUT_PATH, "label_encoder.pkl"))

    def load_pickle(self, file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)

    def preprocess_urls(self, urls: List[str]):
        sequences = self.tokenizer.texts_to_sequences(urls)
        padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        return padded_sequences
