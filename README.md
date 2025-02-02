# This project is deployed on Hugging Face. https://huggingface.co/spaces/rahulbastia/aiResumeScreening
# AI Resume Matcher Project

## Overview
The **AI Resume Matcher** is a Streamlit-based web application designed to analyze resumes and match them against job descriptions using AI and NLP techniques. It extracts relevant skills, experience, job titles, and education details from resumes and computes a similarity score with job descriptions to aid HR professionals in candidate screening.

## Key Features
- **Resume Parsing**: Extracts text from PDF and DOCX files.
- **AI-powered Extraction**: Uses LLM to extract skills, experience, job titles, and education.
- **Similarity Scoring**: Computes cosine similarity between resume and job description embeddings.
- **HR Analytics Dashboard**: Provides insights on skills, education distribution, and candidate ranking.
- **MongoDB Integration**: Stores parsed resumes and shortlisted candidates.
- **Real-time Feedback**: Generates AI-powered suggestions for resume improvement.

## Technologies Used
- **Backend**: Python, MongoDB, Sentence Transformers, SpaCy, PDFMiner, Docx
- **Frontend**: Streamlit, Plotly
- **AI & NLP**: Llama 3.3-70b, Sentence Transformers (MiniLM-L6-v2)
- **DevOps**: Docker (Optional for deployment), MongoDB Atlas, dotenv for environment management

## Architecture
1. **Resume Upload & Parsing**:
   - Extracts text using PDFMiner (PDF) or python-docx (DOCX).
   - Stores extracted text in MongoDB.

2. **AI-driven Analysis**:
   - Processes resume text through Llama 3.3-70b LLM for structured extraction.
   - Converts text to embeddings using Sentence Transformers.
   - Computes similarity with job description embeddings using cosine similarity.

3. **HR Analytics Dashboard**:
   - Displays distribution of top skills and education levels.
   - Provides candidate ranking based on similarity scores.
   - Visualizes hiring trends via interactive graphs.

## Installation & Setup
### Prerequisites
- Python 3.8+
- MongoDB instance (local or MongoDB Atlas)
- Virtual Environment (Recommended)

### Installation Steps
```sh
# Clone the repository
git clone [https://github.com/your-repo/ai-resume-matcher.git](https://github.com/rahulbastia00/rahulbastia00-AI-Powered-Resume-Screening-and-Candidate-Shortlisting.git)
cd ai-resume-matcher

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Update .env with MongoDB and Groq API keys

# Run Streamlit app
streamlit run app.py
```

## Deployment
For deploying on a cloud server or Docker, follow these steps:

### Docker Deployment
```sh
# Build Docker image
docker build -t ai-resume-matcher .

# Run the container
docker run -p 8501:8501 --env-file .env ai-resume-matcher
```

### Cloud Deployment Options
- **Heroku**: Use `heroku container:push web` for deployment.
- **AWS/GCP/Azure**: Deploy via containerized environments.
- **Streamlit Cloud**: Push code to GitHub and deploy via Streamlit Cloud.

## Usage Guide
1. **Upload Resume**: Select a PDF/DOCX file.
2. **Enter Job Description**: Paste the text.
3. **Click 'Analyze'**: The app extracts details and computes a matching score.
4. **Review Results**: View AI-generated feedback, skill insights, and candidate ranking.

## Future Enhancements
- Support for multi-language resume parsing.
- Integration with LinkedIn for real-time job matching.
- Improved AI-driven resume recommendations.

## Contributors
- **Rahul Bastia** - DevOps Engineer & AI Developer
- **Other Contributors**

For any queries, contact [your.email@example.com](mailto:rahul.bastia00@gmail.com).

