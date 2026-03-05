import pandas as pd
import os
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
# import seaborn as sns

DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'dataset')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'etymology_model.pkl')

def load_data(split='train'):
    path = os.path.join(DATASET_DIR, f'{split}.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run clean_dataset.py first.")
    return pd.read_csv(path)

def train_and_evaluate():
    print("Loading data...")
    train_df = load_data('train')
    test_df = load_data('test')
    
    X_train, y_train = train_df['word'], train_df['origin']
    X_test, y_test = test_df['word'], test_df['origin']
    
    print(f"Training on {len(X_train)} samples...")
    
    # Pipeline: Character n-grams (2-4) -> Logistic Regression
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(analyzer='char', ngram_range=(2, 4))),
        ('classifier', LogisticRegression(max_iter=1000, solver='lbfgs'))
    ])
    
    pipeline.fit(X_train, y_train)
    
    print("Evaluating...")
    y_pred = pipeline.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Save model
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == '__main__':
    train_and_evaluate()
