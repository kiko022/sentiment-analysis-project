import os
import argparse
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from datetime import datetime
from data_preprocessing import load_imdb_data, preprocess_data, clean_text


def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def build_model(vocab_size: int, embedding_dim: int = 128, max_len: int = 200):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def main():
    parser = argparse.ArgumentParser(description='IMDB Sentiment Analysis with BiLSTM')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--max_words', type=int, default=10000, help='Maximum number of words')
    parser.add_argument('--max_len', type=int, default=200, help='Maximum sequence length')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    set_seeds(args.seed)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f'logs/{timestamp}'
    model_dir = f'models/{timestamp}'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print("Loading and preprocessing data...")
    train_texts, y_train, test_texts, y_test = load_imdb_data()
    train_texts = [clean_text(text) for text in train_texts]
    test_texts = [clean_text(text) for text in test_texts]
    
    X_train, X_test, tokenizer = preprocess_data(
        train_texts, test_texts, args.max_words, args.max_len
    )
    
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=args.seed
    )
    
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")
    
    print("Building model...")
    model = build_model(args.max_words, embedding_dim=128, max_len=args.max_len)
    
    from tensorflow.keras.optimizers import Adam
    model.compile(
        optimizer=Adam(learning_rate=args.lr),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )
    
    model.summary()
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ModelCheckpoint(f'{model_dir}/best_model.h5', monitor='val_accuracy', save_best_only=True),
        TensorBoard(log_dir=log_dir)
    ]
    
    print("Starting training...")
    history = model.fit(
        X_train, np.array(y_train),
        validation_data=(X_val, np.array(y_val)),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks
    )
    
    print("Evaluating on test set...")
    test_loss, test_acc, test_precision, test_recall = model.evaluate(X_test, np.array(y_test))
    print(f"Test accuracy: {test_acc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
    
    with open(f'{model_dir}/training_log.txt', 'w') as f:
        f.write(f"Test accuracy: {test_acc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}\n")
        f.write(f"Best val accuracy: {max(history.history['val_accuracy']):.4f}\n")
    
    print(f"Model and logs saved to {model_dir}")


if __name__ == "__main__":
    main()
