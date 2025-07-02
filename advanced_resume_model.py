import pandas as pd
import numpy as np
import re
import nltk
import pickle
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class ResumeScreeningModel:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 3),
            sublinear_tf=True,
            min_df=2,
            max_df=0.95
        )
        self.model = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def clean_resume(self, resume_text):
        """Advanced text cleaning function"""
        # Convert to lowercase
        resume_text = resume_text.lower()
        
        # Remove URLs
        resume_text = re.sub(r'http\S+|www.\S+', ' ', resume_text)
        
        # Remove email addresses
        resume_text = re.sub(r'\S+@\S+', ' ', resume_text)
        
        # Remove phone numbers
        resume_text = re.sub(r'\+?\d[\d -]{8,12}\d', ' ', resume_text)
        
        # Remove special characters and numbers but keep some important ones
        resume_text = re.sub(r'[^a-zA-Z\s+#.]', ' ', resume_text)
        
        # Remove extra whitespaces
        resume_text = re.sub(r'\s+', ' ', resume_text)
        
        # Tokenize and lemmatize
        tokens = word_tokenize(resume_text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def load_and_preprocess_data(self, csv_path):
        """Load and preprocess the dataset"""
        print("Loading dataset...")
        df = pd.read_csv(csv_path)
        
        print("Cleaning resumes...")
        df['cleaned_resume'] = df['Resume'].apply(self.clean_resume)
        
        # Encode labels
        df['Category_encoded'] = self.label_encoder.fit_transform(df['Category'])
        
        return df
    
    def create_ensemble_model(self):
        """Create an ensemble model for maximum accuracy"""
        # Individual models
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            class_weight='balanced'
        )
        
        svm = SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            probability=True,
            random_state=42,
            class_weight='balanced'
        )
        
        gb = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        nb = MultinomialNB(alpha=0.1)
        
        lr = LogisticRegression(
            C=10,
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
        
        # Voting ensemble
        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('svm', svm),
                ('gb', gb),
                ('nb', nb),
                ('lr', lr)
            ],
            voting='soft'
        )
        
        return ensemble
    
    def train_model(self, df):
        """Train the model with cross-validation"""
        print("Preparing features...")
        X = df['cleaned_resume']
        y = df['Category_encoded']
        
        # Transform text to TF-IDF features
        X_tfidf = self.tfidf_vectorizer.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print("Creating ensemble model...")
        self.model = self.create_ensemble_model()
        
        print("Training model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"Training Accuracy: {train_score:.4f}")
        print(f"Testing Accuracy: {test_score:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_tfidf, y, cv=5, scoring='accuracy')
        print(f"Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Predictions for detailed evaluation
        y_pred = self.model.predict(X_test)
        
        print("\nDetailed Classification Report:")
        category_names = self.label_encoder.classes_
        print(classification_report(y_test, y_pred, target_names=category_names))
        
        return X_test, y_test, y_pred
    
    def save_model(self, model_path='resume_screening_model.pkl'):
        """Save the trained model and components"""
        model_components = {
            'model': self.model,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'label_encoder': self.label_encoder
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_components, f)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path='resume_screening_model.pkl'):
        """Load a pre-trained model"""
        with open(model_path, 'rb') as f:
            model_components = pickle.load(f)
        
        self.model = model_components['model']
        self.tfidf_vectorizer = model_components['tfidf_vectorizer']
        self.label_encoder = model_components['label_encoder']
        print("Model loaded successfully!")
    
    def predict_resume_category(self, resume_text):
        """Predict the category for a single resume"""
        if self.model is None:
            raise ValueError("Model not trained or loaded!")
        
        # Clean the resume
        cleaned_resume = self.clean_resume(resume_text)
        
        # Transform to TF-IDF
        resume_tfidf = self.tfidf_vectorizer.transform([cleaned_resume])
        
        # Predict
        prediction = self.model.predict(resume_tfidf)[0]
        probability = self.model.predict_proba(resume_tfidf)[0]
        
        # Get category name
        category = self.label_encoder.inverse_transform([prediction])[0]
        confidence = max(probability)
        
        # Get top 3 predictions
        top_3_indices = np.argsort(probability)[-3:][::-1]
        top_3_categories = self.label_encoder.inverse_transform(top_3_indices)
        top_3_probs = probability[top_3_indices]
        
        return {
            'predicted_category': category,
            'confidence': confidence,
            'top_3_predictions': list(zip(top_3_categories, top_3_probs))
        }

def main():
    # Initialize the model
    resume_model = ResumeScreeningModel()
    
    # Load and preprocess data
    df = resume_model.load_and_preprocess_data('UpdatedResumeDataSet.csv')
    
    # Train the model
    X_test, y_test, y_pred = resume_model.train_model(df)
    
    # Save the model
    resume_model.save_model()
    
    # Test with a sample resume
    sample_resume = """
    Experienced Python developer with 5 years of experience in machine learning and data science.
    Proficient in scikit-learn, pandas, numpy, matplotlib. Built multiple ML models for classification
    and regression problems. Experience with Flask, Django for web development.
    """
    
    result = resume_model.predict_resume_category(sample_resume)
    print(f"\nSample Prediction:")
    print(f"Category: {result['predicted_category']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Top 3 predictions: {result['top_3_predictions']}")

if __name__ == "__main__":
    main()
