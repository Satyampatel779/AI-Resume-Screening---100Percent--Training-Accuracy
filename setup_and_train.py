"""
Complete Resume Screening System Setup and Training Script
This script will train the model and prepare the deployment
"""

import sys
import os
import subprocess

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("✅ Packages installed successfully!")

def train_model():
    """Train the resume screening model"""
    print("Starting model training...")
    
    # Import after installation
    from advanced_resume_model import ResumeScreeningModel
    
    # Initialize the model
    model = ResumeScreeningModel()
    
    # Check if dataset exists
    if not os.path.exists('UpdatedResumeDataSet.csv'):
        print("❌ Error: UpdatedResumeDataSet.csv not found!")
        print("Please ensure the dataset file is in the current directory.")
        return False
    
    # Load and preprocess data
    print("Loading and preprocessing dataset...")
    df = model.load_and_preprocess_data('UpdatedResumeDataSet.csv')
    print(f"Dataset loaded: {df.shape[0]} resumes across {df['Category'].nunique()} categories")
    
    # Train the model
    print("Training ensemble model...")
    X_test, y_test, y_pred = model.train_model(df)
    
    # Save the model
    print("Saving trained model...")
    model.save_model('resume_screening_model.pkl')
    
    # Test the model with a sample
    print("\n" + "="*50)
    print("TESTING THE MODEL")
    print("="*50)
    
    sample_resumes = [
        "Python developer with machine learning experience using scikit-learn, pandas, numpy",
        "Java developer with Spring Boot and microservices experience",
        "HR professional with recruitment and talent management experience",
        "Data scientist with deep learning and neural networks expertise"
    ]
    
    for i, sample in enumerate(sample_resumes, 1):
        result = model.predict_resume_category(sample)
        print(f"\nSample {i}: {sample[:50]}...")
        print(f"Predicted: {result['predicted_category']} (Confidence: {result['confidence']:.3f})")
    
    print("\n✅ Model training completed successfully!")
    return True

def setup_flask_app():
    """Setup Flask application"""
    print("\n" + "="*50)
    print("FLASK APP SETUP")
    print("="*50)
    
    # Create uploads directory
    os.makedirs('uploads', exist_ok=True)
    print("✅ Uploads directory created")
    
    # Check if all required files exist
    required_files = [
        'app.py',
        'templates/index.html',
        'templates/result.html',
        'resume_screening_model.pkl'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required files are present")
    return True

def main():
    print("🚀 AI Resume Screening System Setup")
    print("="*50)
    
    try:
        # Step 1: Install requirements
        install_requirements()
        
        # Step 2: Train the model
        if not train_model():
            print("❌ Model training failed!")
            return
        
        # Step 3: Setup Flask app
        if not setup_flask_app():
            print("❌ Flask app setup failed!")
            return
        
        print("\n" + "="*50)
        print("🎉 SETUP COMPLETED SUCCESSFULLY!")
        print("="*50)
        print("\nTo start the web application, run:")
        print("python app.py")
        print("\nThen open your browser and go to:")
        print("http://localhost:5000")
        print("\n📋 Features:")
        print("- Upload PDF, DOCX, or TXT resume files")
        print("- Paste resume text directly")
        print("- Get instant category predictions")
        print("- View confidence scores")
        print("- See top 3 predictions")
        print("- Download results as JSON")
        
    except Exception as e:
        print(f"❌ Error during setup: {str(e)}")
        print("Please check the error and try again.")

if __name__ == "__main__":
    main()
