# app.py
import streamlit as st
import os
import re
import spacy
import pandas as pd
import plotly.express as px
from pdfminer.high_level import extract_text
from docx import Document
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize components
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("all-MiniLM-L6-v2")

# MongoDB connection
client = MongoClient(os.getenv("MONGO_DB"))
db = client["resume_database"]

def extract_text_from_pdf(uploaded_file):
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    return extract_text("temp.pdf")

def extract_text_from_docx(uploaded_file):
    with open("temp.docx", "wb") as f:
        f.write(uploaded_file.getbuffer())
    doc = Document("temp.docx")
    return "\t".join([para.text for para in doc.paragraphs])

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop])

def store_resume(data):
    collection = db["resumes"]
    collection.insert_one(data)

def get_groq_response(prompt):
    llm = ChatGroq(
        model_name="llama3-70b-8192",
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    return llm.invoke(prompt).content

# Streamlit UI
st.title("AI Resume Matcher 🧑💼")
st.subheader("Upload Resume and Job Description")

# File upload section
col1, col2 = st.columns(2)
with col1:
    resume_file = st.file_uploader("Upload Resume", type=["pdf", "docx"])
with col2:
    job_desc = st.text_area("Paste Job Description", height=200)

if st.button("Analyze"):
    if resume_file and job_desc:
        with st.spinner("Processing..."):
            # Process resume
            if resume_file.name.endswith('.pdf'):
                raw_text = extract_text_from_pdf(resume_file)
            else:
                raw_text = extract_text_from_docx(resume_file)
            
            cleaned_text = preprocess_text(raw_text)
            
            # Store in MongoDB
            resume_data = {
                "name": resume_file.name,
                "resume_text": cleaned_text,
                "status": "unprocessed"
            }
            store_resume(resume_data)
            
            # Generate embeddings
            resume_embedding = model.encode(cleaned_text).tolist()
            job_embedding = model.encode(job_desc).tolist()
            
            # Calculate similarity
            similarity = cosine_similarity([resume_embedding], [job_embedding])[0][0]
            
            # Generate feedback
            feedback_prompt = f"""
            Resume: {cleaned_text}
            Job Description: {job_desc}
            
            Generate detailed analysis with:
            1. Top 3 matching qualifications
            2. Missing skills
            3. Improvement suggestions
            4. Overall suitability score ({similarity:.2f}/1.00)
            """
            analysis = get_groq_response(feedback_prompt)
            
            # Display results
            st.success("Analysis Complete!")
            st.subheader(f"Matching Score: {similarity:.2%}")
            
            st.markdown("### Detailed Analysis")
            st.write(analysis)
            
            # Visualization
            st.subheader("Skill Distribution")
            doc = nlp(cleaned_text)
            skills = [ent.text for ent in doc.ents if ent.label_ == "SKILL"]
            if skills:
                skill_df = pd.DataFrame(skills, columns=["Skill"]).value_counts().reset_index()
                fig = px.bar(skill_df, x="Skill", y="count", title="Top Skills in Resume")
                st.plotly_chart(fig)
            else:
                st.warning("No skills detected in resume")
            
    else:
        st.error("Please upload both resume and job description")

# Analytics Dashboard
st.sidebar.header("HR Analytics")
if st.sidebar.button("Show Dashboard"):
    with st.spinner("Loading Analytics..."):
        # Fetch data
        resumes = list(db["resumes"].find())
        shortlisted = list(db["shortlisted_candidates"].find())
        
        if resumes:
            # Skill distribution
            st.subheader("Skill Distribution Across Resumes")
            all_skills = [resume.get('skills', []) for resume in resumes]
            flat_skills = [skill for sublist in all_skills for skill in sublist]
            if flat_skills:
                skill_counts = pd.Series(flat_skills).value_counts().reset_index()
                skill_counts.columns = ["Skill", "Count"]
                fig = px.bar(skill_counts.head(10), x="Skill", y="Count")
                st.plotly_chart(fig)
            
            # Similarity scores
            if shortlisted:
                st.subheader("Candidate Rankings")
                df = pd.DataFrame(shortlisted)
                fig = px.bar(df, x="name", y="score", color="score")
                st.plotly_chart(fig)

# Instructions
st.sidebar.markdown("""
**Instructions:**
1. Upload resume (PDF/DOCX)
2. Paste job description
3. Click Analyze
4. View results & analytics
""")