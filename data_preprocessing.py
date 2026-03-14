import re
import string
from typing import Tuple, List
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_imdb_data() -> Tuple[List[str], List[int], List[str], List[int]]:
    from tensorflow.keras.datasets import imdb
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
    word_index = imdb.get_word_index()
    reverse_word_index = {v: k for k, v in word_index.items()}
    
    def decode_review(encoded_review):
        return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])
    
    train_texts = [decode_review(x) for x in x_train]
    test_texts = [decode_review(x) for x in x_test]
    
    return train_texts, y_train, test_texts, y_test


def preprocess_data(
    train_texts: List[str],
    test_texts: List[str],
    max_words: int = 10000,
    max_len: int = 200
) -> Tuple[np.ndarray, np.ndarray, Tokenizer]:
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(train_texts)
    
    train_sequences = tokenizer.texts_to_sequences(train_texts)
    test_sequences = tokenizer.texts_to_sequences(test_texts)
    
    train_padded = pad_sequences(train_sequences, maxlen=max_len, padding='post', truncating='post')
    test_padded = pad_sequences(test_sequences, maxlen=max_len, padding='post', truncating='post')
    
    return train_padded, test_padded, tokenizer
