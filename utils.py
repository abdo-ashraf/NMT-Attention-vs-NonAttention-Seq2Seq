import unicodedata
import pyarabic.araby as araby
import contractions
import re
from camel_tools.tokenizers.word import simple_word_tokenize
import spacy

# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def preprocess_ar(text):
    text = text.lower()
    text = unicodeToAscii(text)
    text = re.sub(r"([?.!؟،,¿])", r" \1 ", text)
    text = re.sub(r"[^؀-ۿ?.!,¿]+", " ", text)
    text = araby.strip_diacritics(text) # Remove diacritics "التشكيل"
    text = re.sub(r'\s+', ' ', text).strip() # Trim multiple whitespaces to one
    return text

def preprocess_en(text):
    text = text.lower()
    text = contractions.fix(text) # Fix contractions "it's" -> "it is"
    text = re.sub(r"([?.!,¿])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text)
    text = re.sub(r'\s+', ' ', text).strip() # Trim multiple whitespaces to one
    return text

# Arabic Tokenizer
class camel_tokenizer():
    def __call__(self, text):
        return simple_word_tokenize(text)
        # return [tok.text.lower() for tok in simple_word_tokenize(text)]


# English Tokenizer
class spacy_tokenizer:
    def __init__(self):
        self.spacy_eng = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        self.spacy_eng.max_length = 10**6

    def __call__(self, text):
        return [tok.text for tok in self.spacy_eng.tokenizer(text)]

    def batch_tokenize(self, texts):
        return [[tok.text for tok in doc] for doc in self.spacy_eng.pipe(texts, batch_size=256)]