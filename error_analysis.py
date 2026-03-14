import numpy as np
import tensorflow as tf
from data_preprocessing import load_imdb_data, preprocess_data, clean_text
from train import build_model


def analyze_errors():
    print("Loading data...")
    train_texts, y_train, test_texts, y_test = load_imdb_data()
    test_texts_clean = [clean_text(text) for text in test_texts]
    
    X_train, X_test, tokenizer = preprocess_data(train_texts, test_texts, 10000, 200)
    
    print("Loading model...")
    model = build_model(10000, 128, 200)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print("Making predictions...")
    predictions = model.predict(X_test)
    pred_classes = (predictions > 0.5).astype(int).flatten()
    
    print("\n=== Error Analysis ===")
    false_positives = []
    false_negatives = []
    
    for i in range(len(y_test)):
        if pred_classes[i] != y_test[i]:
            if pred_classes[i] == 1:
                false_positives.append((test_texts[i], predictions[i][0]))
            else:
                false_negatives.append((test_texts[i], predictions[i][0]))
    
    print(f"\nFalse Positives (predicted positive, actually negative): {len(false_positives)}")
    print(f"False Negatives (predicted negative, actually positive): {len(false_negatives)}")
    
    print("\n--- Example False Negative ---")
    if false_negatives:
        print(f"Review: {false_negatives[0][0][:300]}...")
        print(f"Prediction probability: {false_negatives[0][1]:.4f}")
    
    print("\n--- Example False Positive ---")
    if false_positives:
        print(f"Review: {false_positives[0][0][:300]}...")
        print(f"Prediction probability: {false_positives[0][1]:.4f}")
    
    with open('logs/error_analysis.txt', 'w', encoding='utf-8') as f:
        f.write("=== Error Analysis Report ===\n\n")
        f.write(f"False Positives: {len(false_positives)}\n")
        f.write(f"False Negatives: {len(false_negatives)}\n\n")
        
        f.write("\n--- False Negatives ---\n")
        for i, (text, prob) in enumerate(false_negatives[:5]):
            f.write(f"\nExample {i+1} (prob: {prob:.4f}):\n")
            f.write(text[:500] + "\n")
        
        f.write("\n--- False Positives ---\n")
        for i, (text, prob) in enumerate(false_positives[:5]):
            f.write(f"\nExample {i+1} (prob: {prob:.4f}):\n")
            f.write(text[:500] + "\n")
    
    print("\nError analysis saved to logs/error_analysis.txt")


if __name__ == "__main__":
    analyze_errors()
