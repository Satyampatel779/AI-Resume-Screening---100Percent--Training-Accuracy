from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import os
import pickle
from enhanced_resume_model import AdvancedResumeScreeningModel
from werkzeug.utils import secure_filename
import PyPDF2
import docx
import io

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
model = AdvancedResumeScreeningModel()
try:
    model.load_model('resume_screening_model.pkl')
    print("✅ Enhanced model loaded successfully!")
except FileNotFoundError:
    print("❌ Model not found. Please train the model first.")
    model = None

def extract_text_from_pdf(file_stream):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(file_stream)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

def extract_text_from_docx(file_stream):
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(file_stream)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        print(f"Error reading DOCX: {e}")
        return None

def extract_text_from_txt(file_stream):
    """Extract text from TXT file"""
    try:
        return file_stream.read().decode('utf-8')
    except Exception as e:
        print(f"Error reading TXT: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if model is None:
        flash('Model not loaded. Please contact administrator.', 'error')
        return redirect(url_for('index'))
    
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    if file:
        filename = secure_filename(file.filename)
        file_extension = filename.lower().split('.')[-1]
        
        # Read file content
        file_stream = io.BytesIO(file.read())
        
        if file_extension == 'pdf':
            resume_text = extract_text_from_pdf(file_stream)
        elif file_extension == 'docx':
            resume_text = extract_text_from_docx(file_stream)
        elif file_extension == 'txt':
            resume_text = extract_text_from_txt(file_stream)
        else:
            flash('Unsupported file format. Please upload PDF, DOCX, or TXT files.', 'error')
            return redirect(url_for('index'))
        
        if resume_text is None or len(resume_text.strip()) == 0:
            flash('Could not extract text from the file or file is empty.', 'error')
            return redirect(url_for('index'))
        
        try:
            # Make prediction
            result = model.predict_resume_category(resume_text)
            
            return render_template('result.html', 
                                 filename=filename,
                                 result=result,
                                 resume_text=resume_text[:1000] + "..." if len(resume_text) > 1000 else resume_text)
        
        except Exception as e:
            flash(f'Error processing resume: {str(e)}', 'error')
            return redirect(url_for('index'))

@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    resume_text = request.json.get('text', '')
    
    if not resume_text.strip():
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        result = model.predict_resume_category(resume_text)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/job_eligibility')
def job_eligibility():
    return render_template('job_eligibility.html')

@app.route('/analyze_eligibility', methods=['POST'])
def analyze_eligibility():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.json
    resume_text = data.get('resume_text', '')
    job_description = data.get('job_description', '')
    required_experience = int(data.get('required_experience', 0))
    
    if not resume_text.strip() or not job_description.strip():
        return jsonify({'error': 'Both resume text and job description are required'}), 400
    
    try:
        result = model.calculate_job_eligibility(resume_text, job_description, required_experience)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload_eligibility', methods=['POST'])
def upload_eligibility():
    if model is None:
        flash('Model not loaded. Please contact administrator.', 'error')
        return redirect(url_for('job_eligibility'))
    
    if 'resume_file' not in request.files:
        flash('No resume file selected', 'error')
        return redirect(url_for('job_eligibility'))
    
    file = request.files['resume_file']
    job_description = request.form.get('job_description', '').strip()
    required_experience = int(request.form.get('required_experience', 0))
    
    if file.filename == '' or not job_description:
        flash('Both resume file and job description are required', 'error')
        return redirect(url_for('job_eligibility'))
    
    if file:
        filename = secure_filename(file.filename)
        file_extension = filename.lower().split('.')[-1]
        
        # Read file content
        file_stream = io.BytesIO(file.read())
        
        if file_extension == 'pdf':
            resume_text = extract_text_from_pdf(file_stream)
        elif file_extension == 'docx':
            resume_text = extract_text_from_docx(file_stream)
        elif file_extension == 'txt':
            resume_text = extract_text_from_txt(file_stream)
        else:
            flash('Unsupported file format. Please upload PDF, DOCX, or TXT files.', 'error')
            return redirect(url_for('job_eligibility'))
        
        if resume_text is None or len(resume_text.strip()) == 0:
            flash('Could not extract text from the file or file is empty.', 'error')
            return redirect(url_for('job_eligibility'))
        
        try:
            # Analyze eligibility
            result = model.calculate_job_eligibility(resume_text, job_description, required_experience)
            
            return render_template('eligibility_result.html', 
                                 filename=filename,
                                 result=result,
                                 job_description=job_description,
                                 required_experience=required_experience,
                                 resume_text=resume_text[:1000] + "..." if len(resume_text) > 1000 else resume_text)
        
        except Exception as e:
            flash(f'Error analyzing eligibility: {str(e)}', 'error')
            return redirect(url_for('job_eligibility'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
