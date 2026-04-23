import streamlit as st
import PyPDF2
from groq import Groq
import os
from dotenv import load_dotenv
import re

# -------------------- CONFIG --------------------
st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄")

# -------------------- LOAD ENV --------------------
load_dotenv(dotenv_path=".env")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -------------------- PDF EXTRACTION --------------------
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted
    return text.lower()

# -------------------- CLEAN TEXT --------------------
def clean_text(text):
    stopwords = set([
        "and","or","the","is","in","at","of","for","to","a","an","on","with","by","as"
    ])
    words = re.findall(r'\b\w+\b', text.lower())
    return [w for w in words if w not in stopwords]

# -------------------- KEYWORD MATCHING --------------------
def keyword_matching(resume_text, job_desc):
    jd_words = clean_text(job_desc)
    keywords = list(set(jd_words))

    matched = [word for word in keywords if word in resume_text]
    score = int((len(matched) / len(keywords)) * 100) if keywords else 0

    return score, keywords, matched

# -------------------- SKILL EXTRACTION --------------------
def extract_skills(text):
    skills_list = [
        "python","machine learning","deep learning","tensorflow","pytorch",
        "nlp","docker","kubernetes","sql","power bi","gcp","aws","azure",
        "data analysis","ci/cd","devops","linux","spark","hadoop"
    ]

    found = []
    for skill in skills_list:
        if skill in text:
            found.append(skill)

    return found, skills_list

# -------------------- AI FUNCTION --------------------
def analyze_resume_with_ai(resume_text, job_desc):
    prompt = f"""
    You are an ATS system.

    Compare resume with job description.

    RULES:
    - Only plain text
    - Match Score in %
    - No extra sections

    FORMAT:

    Match Score: XX%

    Missing Skills:
    - skill 1
    - skill 2

    Suggestions:
    - suggestion 1
    - suggestion 2

    Resume:
    {resume_text}

    Job Description:
    {job_desc}
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# -------------------- CLEAN OUTPUT --------------------
def clean_output(text):
    text = text.replace("**", "")
    text = re.sub(r"Resume:.*", "", text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()

# -------------------- UI --------------------
st.title("📄 AI Resume Analyzer")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
job_desc = st.text_area("Paste Job Description")

# Resume Preview
if uploaded_file:
    preview = extract_text_from_pdf(uploaded_file)
    with st.expander("📄 Resume Preview"):
        st.text_area("Preview", preview[:1000], height=200)

# -------------------- MAIN --------------------
if st.button("Analyze"):

    if uploaded_file and job_desc:

        resume_text = extract_text_from_pdf(uploaded_file)

        # -------- Keyword Matching --------
        keyword_score, keywords, matched = keyword_matching(resume_text, job_desc)

        st.subheader("Keyword Matching Score")
        st.metric("Keyword Score", f"{keyword_score}%")
        st.progress(keyword_score)

        # -------- Skill Matching --------
        st.subheader("Skill Match Analysis")

        found_skills, all_skills = extract_skills(resume_text)

        for skill in all_skills:
            if skill in found_skills:
                st.write(f"✅ {skill}")
            else:
                st.write(f"❌ {skill}")

        # -------- AI Analysis --------
        with st.spinner("Analyzing with AI..."):
            result = analyze_resume_with_ai(resume_text, job_desc)

        result = clean_output(result)

        score_match = re.search(r'(\d+)%', result)

        if score_match:
            score_value = int(score_match.group(1))
            st.subheader("AI Match Score")
            st.metric("AI Score", f"{score_value}%")
            st.progress(score_value)

        st.divider()

        # -------- Extract Sections --------
        missing_skills = re.search(
            r'Missing Skills:(.*?)(Suggestions:|$)',
            result,
            re.DOTALL | re.IGNORECASE
        )

        suggestions = re.search(
            r'Suggestions:(.*)',
            result,
            re.DOTALL | re.IGNORECASE
        )

        st.subheader("Analysis Result")

        if missing_skills:
            st.markdown("### Missing Skills")
            st.markdown(missing_skills.group(1).strip())

        if suggestions:
            st.markdown("### Suggestions")
            st.markdown(suggestions.group(1).strip())

        # -------- DOWNLOAD REPORT --------
        report = f"""
AI Resume Analyzer Report

Keyword Score: {keyword_score}%
AI Score: {score_value if score_match else "N/A"}%

Skills Found:
{", ".join(found_skills)}

Missing Skills:
{missing_skills.group(1).strip() if missing_skills else ""}

Suggestions:
{suggestions.group(1).strip() if suggestions else ""}
"""

        st.download_button(
            label="📥 Download Report",
            data=report,
            file_name="resume_analysis.txt",
            mime="text/plain"
        )

    else:
        st.warning("Please upload resume and job description")