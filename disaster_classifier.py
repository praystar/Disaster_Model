from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

class DisasterClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('clf', MultinomialNB())
        ])
        
    def train(self, X, y):
        """
        Train the classifier with disaster text data
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.pipeline.fit(X_train, y_train)
        
        # Calculate accuracy
        accuracy = self.pipeline.score(X_test, y_test)
        return accuracy
    
    def predict(self, texts):
        """
        Predict disaster type and severity for new texts
        """
        return self.pipeline.predict(texts)
    
    def save_model(self, filepath):
        """
        Save the trained model
        """
        joblib.dump(self.pipeline, filepath)
    
    @classmethod
    def load_model(cls, filepath):
        """
        Load a trained model
        """
        instance = cls()
        instance.pipeline = joblib.load(filepath)
        return instance 