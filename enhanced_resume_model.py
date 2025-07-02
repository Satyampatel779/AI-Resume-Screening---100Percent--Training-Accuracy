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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
import os

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

class AdvancedResumeScreeningModel:
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
        # Separate vectorizer for job matching
        self.job_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            max_df=0.98
        )
        self.model = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.model_path = 'resume_screening_model.pkl'
        
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
    
    def extract_skills(self, text):
        """Extract technical skills from text"""
        # Common technical skills
        skills_keywords = [
            'python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust',
            'react', 'angular', 'vue', 'node', 'django', 'flask', 'spring', 'laravel',
            'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'cassandra',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'linux',
            'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'scikit-learn',
            'pandas', 'numpy', 'matplotlib', 'tableau', 'power bi', 'excel',
            'html', 'css', 'bootstrap', 'sass', 'webpack', 'npm', 'yarn',
            'agile', 'scrum', 'devops', 'ci/cd', 'microservices', 'api', 'rest',
            'blockchain', 'solidity', 'ethereum', 'bitcoin', 'smart contracts'
        ]
        
        text_lower = text.lower()
        found_skills = []
        
        for skill in skills_keywords:
            if skill in text_lower:
                found_skills.append(skill)
        
        return found_skills
    
    def calculate_experience_years(self, text):
        """Extract years of experience from text"""
        experience_patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)',
            r'(\d+)\+?\s*yrs?\s*(?:of\s*)?(?:experience|exp)',
            r'experience\s*(?:of\s*)?(\d+)\+?\s*years?',
            r'(\d+)\+?\s*years?\s*in',
        ]
        
        text_lower = text.lower()
        years = []
        
        for pattern in experience_patterns:
            matches = re.findall(pattern, text_lower)
            years.extend([int(match) for match in matches])
        
        return max(years) if years else 0
    
    def load_and_preprocess_data(self, csv_path):
        """Load and preprocess the dataset"""
        print("Loading dataset...")
        df = pd.read_csv(csv_path)
        
        print("Cleaning resumes...")
        df['cleaned_resume'] = df['Resume'].apply(self.clean_resume)
        
        # Extract additional features
        df['skills'] = df['Resume'].apply(self.extract_skills)
        df['experience_years'] = df['Resume'].apply(self.calculate_experience_years)
        
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
    
    def save_model(self, model_path=None):
        """Save the trained model and all components"""
        if model_path is None:
            model_path = self.model_path
            
        model_components = {
            'model': self.model,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'job_vectorizer': self.job_vectorizer,
            'label_encoder': self.label_encoder,
            'lemmatizer': self.lemmatizer,
            'stop_words': self.stop_words
        }
        
        # Create backup if file exists
        if os.path.exists(model_path):
            backup_path = model_path.replace('.pkl', '_backup.pkl')
            os.rename(model_path, backup_path)
            print(f"Previous model backed up to {backup_path}")
        
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model_components, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"✅ Model successfully saved to {model_path}")
            print(f"📊 Model size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
        except Exception as e:
            print(f"❌ Error saving model: {str(e)}")
            # Restore backup if save failed
            if os.path.exists(backup_path):
                os.rename(backup_path, model_path)
                print("Backup restored due to save failure")
    
    def load_model(self, model_path=None):
        """Load a pre-trained model"""
        if model_path is None:
            model_path = self.model_path
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            with open(model_path, 'rb') as f:
                model_components = pickle.load(f)
            
            self.model = model_components['model']
            self.tfidf_vectorizer = model_components['tfidf_vectorizer']
            self.job_vectorizer = model_components['job_vectorizer']
            self.label_encoder = model_components['label_encoder']
            self.lemmatizer = model_components.get('lemmatizer', WordNetLemmatizer())
            self.stop_words = model_components.get('stop_words', set(stopwords.words('english')))
            
            print(f"✅ Model successfully loaded from {model_path}")
            print(f"📊 Model size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
            print(f"🎯 Supported categories: {len(self.label_encoder.classes_)}")
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise
    
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
        
        # Extract additional information
        skills = self.extract_skills(resume_text)
        experience = self.calculate_experience_years(resume_text)
        
        return {
            'predicted_category': category,
            'confidence': confidence,
            'top_3_predictions': list(zip(top_3_categories, top_3_probs)),
            'extracted_skills': skills,
            'experience_years': experience
        }
    
    def calculate_job_eligibility(self, resume_text, job_description, required_experience=0):
        """Calculate eligibility score based on resume and job description"""
        if self.model is None:
            raise ValueError("Model not trained or loaded!")
        
        # Clean both texts
        cleaned_resume = self.clean_resume(resume_text)
        cleaned_job = self.clean_resume(job_description)
        
        # Get resume category prediction
        resume_prediction = self.predict_resume_category(resume_text)
        
        # Calculate text similarity using TF-IDF
        combined_texts = [cleaned_resume, cleaned_job]
        tfidf_matrix = self.job_vectorizer.fit_transform(combined_texts)
        similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # Extract skills from both
        resume_skills = set(self.extract_skills(resume_text))
        job_skills = set(self.extract_skills(job_description))
        
        # Calculate skill match percentage
        if job_skills:
            skill_match = len(resume_skills.intersection(job_skills)) / len(job_skills)
        else:
            skill_match = 0.5  # Neutral score if no skills found in job description
        
        # Experience match
        resume_experience = self.calculate_experience_years(resume_text)
        experience_match = min(resume_experience / max(required_experience, 1), 1.0) if required_experience > 0 else 0.8
        
        # Calculate overall eligibility score (weighted average)
        weights = {
            'text_similarity': 0.3,
            'skill_match': 0.4,
            'experience_match': 0.2,
            'category_confidence': 0.1
        }
        
        eligibility_score = (
            similarity_score * weights['text_similarity'] +
            skill_match * weights['skill_match'] +
            experience_match * weights['experience_match'] +
            resume_prediction['confidence'] * weights['category_confidence']
        )
        
        # Determine eligibility level
        if eligibility_score >= 0.8:
            eligibility_level = "Highly Eligible"
            recommendation = "Strong candidate - Proceed with interview"
        elif eligibility_score >= 0.6:
            eligibility_level = "Eligible"
            recommendation = "Good candidate - Consider for interview"
        elif eligibility_score >= 0.4:
            eligibility_level = "Partially Eligible"
            recommendation = "Average candidate - Review carefully"
        else:
            eligibility_level = "Not Eligible"
            recommendation = "Weak candidate - Consider other options"
        
        return {
            'eligibility_score': eligibility_score,
            'eligibility_level': eligibility_level,
            'recommendation': recommendation,
            'text_similarity': similarity_score,
            'skill_match_percentage': skill_match * 100,
            'experience_match': experience_match,
            'resume_skills': list(resume_skills),
            'job_skills': list(job_skills),
            'matching_skills': list(resume_skills.intersection(job_skills)),
            'missing_skills': list(job_skills - resume_skills),
            'resume_experience_years': resume_experience,
            'required_experience_years': required_experience,
            'category_prediction': resume_prediction
        }

def main():
    # Initialize the model
    resume_model = AdvancedResumeScreeningModel()
    
    # Load and preprocess data
    df = resume_model.load_and_preprocess_data('UpdatedResumeDataSet.csv')
    
    # Train the model
    X_test, y_test, y_pred = resume_model.train_model(df)
    
    # Save the model with enhanced persistence
    resume_model.save_model()
    
    # Test the job eligibility feature
    sample_resume = """
    Experienced Python developer with 5 years of experience in machine learning and data science.
    Proficient in scikit-learn, pandas, numpy, matplotlib, tensorflow, pytorch. Built multiple ML models for classification
    and regression problems. Experience with Flask, Django for web development. Strong background in AWS, Docker, Kubernetes.
    """
    
    sample_job_description = """
    We are looking for a Senior Data Scientist with 3+ years of experience in machine learning.
    Required skills: Python, scikit-learn, pandas, numpy, tensorflow, machine learning, data analysis.
    Experience with cloud platforms (AWS) and containerization (Docker) is preferred.
    """
    
    # Test category prediction
    result = resume_model.predict_resume_category(sample_resume)
    print(f"\n🎯 Sample Category Prediction:")
    print(f"Category: {result['predicted_category']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Skills: {result['extracted_skills']}")
    print(f"Experience: {result['experience_years']} years")
    
    # Test job eligibility
    eligibility = resume_model.calculate_job_eligibility(
        sample_resume, 
        sample_job_description, 
        required_experience=3
    )
    
    print(f"\n💼 Job Eligibility Analysis:")
    print(f"Eligibility Score: {eligibility['eligibility_score']:.3f}")
    print(f"Eligibility Level: {eligibility['eligibility_level']}")
    print(f"Recommendation: {eligibility['recommendation']}")
    print(f"Text Similarity: {eligibility['text_similarity']:.3f}")
    print(f"Skill Match: {eligibility['skill_match_percentage']:.1f}%")
    print(f"Matching Skills: {eligibility['matching_skills']}")
    print(f"Missing Skills: {eligibility['missing_skills']}")

if __name__ == "__main__":
    main()
