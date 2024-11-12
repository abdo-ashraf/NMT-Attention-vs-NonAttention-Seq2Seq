from utils import spacy_tokenizer
from collections import Counter

class Vocabulary:
    def __init__(self, tokenizer, max_freq=3, unk=True, sos=False, eos=False):
        self.sos = sos
        self.eos = eos
        self.unk = unk
        self.tokenizer = tokenizer
        self.max_freq = max_freq

        self.stoi = {"<PAD>": 0}
        if unk: self.stoi['<UNK>'] = len(self.stoi)
        if sos: self.stoi['<SOS>'] = len(self.stoi)
        if eos: self.stoi['<EOS>'] = len(self.stoi)

    def __len__(self):
        return len(self.stoi)

    def get_vocabulary(self):
        return self.stoi

    def set_vocabulary(self, stoi):
        self.stoi = stoi

    def add_token(self, token_name: str):
        if token_name not in self.stoi:
            self.stoi[token_name] = len(self.stoi)

    def build_vocabulary(self, sentences_list):
        if isinstance(sentences_list[0], list):
            sentences_list = [' '.join(sentence) for sentence in sentences_list]

        if isinstance(self.tokenizer, spacy_tokenizer) and hasattr(self.tokenizer, 'batch_tokenize'):
            tokens_list = self.tokenizer.batch_tokenize(sentences_list)
        else:
            tokens_list = [self.tokenizer(sentence) for sentence in sentences_list]

        word_counts = Counter(token for tokens in tokens_list for token in tokens)
        filtered_words = [word for word, count in word_counts.items() if count >= self.max_freq]
        self.stoi.update({word: i+len(self.stoi) for i, word in enumerate(filtered_words)})

    def get_numerical_tokens(self, text):
        tokens = self.tokenizer(text)
        if self.sos: tokens.insert(0, '<SOS>')
        if self.eos: tokens.append('<EOS>')
        unk_id = self.stoi.get('<UNK>', None)
        return [self.stoi.get(token, unk_id) for token in tokens]

    def __call__(self, text):
        return self.get_numerical_tokens(text)

    def tokens_to_text(self, tokens_list):
        keys = list(self.stoi.keys())
        values = list(self.stoi.values())

        return ' '.join([keys[values.index(token)] for token in tokens_list])