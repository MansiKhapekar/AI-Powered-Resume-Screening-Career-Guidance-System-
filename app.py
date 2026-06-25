import streamlit as st
import joblib
import pdfplumber

st.set_page_config(page_title="AI Interview Preparation System")

st.markdown("""
<h1 style='text-align:center;color:#4F46E5;'>
 AI-Powered Resume Screening & Career Guidance System
</h1>
""", unsafe_allow_html=True)

model = joblib.load("rf_model.pkl")

st.success("Model Loaded Successfully!")

uploaded_file = st.file_uploader(
    "Upload Your Resume (PDF)",
    type=["pdf"]
)

if uploaded_file:

    text = ""

    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    st.subheader("Resume Text")

    st.text_area(
        "Extracted Resume",
        text,
        height=300
    )

    # Skill Extraction

resume_lower = text.lower()

python_skill = 1 if ("python" in resume_lower) else 0
sql_skill = 1 if "sql" in resume_lower else 0
ml_skill = 1 if "machine learning" in resume_lower else 0
powerbi_skill = 1 if "power bi" in resume_lower else 0
aws_skill = 1 if "aws" in resume_lower else 0
java_skill = 1 if "java" in resume_lower else 0

cpp_skill = 1 if (
    "c++" in resume_lower or
    "cpp" in resume_lower
) else 0

data_analytics_skill = 1 if (
    "data analytics" in resume_lower or
    "data analysis" in resume_lower
) else 0



st.subheader("Detected Skills")

st.write("Python:", python_skill)
st.write("SQL:", sql_skill)
st.write("Machine Learning:", ml_skill)
st.write("Power BI:", powerbi_skill)
st.write("AWS:", aws_skill)
st.write("Java:", java_skill)
st.write("C++:", cpp_skill)
st.write("Data Analytics:", data_analytics_skill)


user_data = [[
    python_skill,
    sql_skill,
    ml_skill,
    powerbi_skill,
    aws_skill,
    java_skill,
    cpp_skill,
    data_analytics_skill
]]

predicted_role = model.predict(user_data)

st.subheader("Predicted Job Role")

st.success(predicted_role[0])

# Prediction Probability

probabilities = model.predict_proba(user_data)[0]

roles = model.classes_

result = list(zip(roles, probabilities))

result = sorted(
    result,
    key=lambda x: x[1],
    reverse=True
)

st.subheader("Top 3 Suitable Roles")

for role, prob in result[:3]:
    st.write(
        f"{role} : {round(prob*100,2)} %"
    )


skills_found = (
    python_skill +
    sql_skill +
    ml_skill +
    powerbi_skill +
    aws_skill +
    java_skill +
    cpp_skill +
    data_analytics_skill
)

resume_score = (skills_found / 8) * 100

st.subheader("Resume Score")

st.progress(int(resume_score))

st.success(
    f"Resume Score : {round(resume_score,2)} / 100"
)
st.metric(
    label="Resume Score",
    value=f"{round(resume_score,2)}%"
)

import google.generativeai as genai

genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

if st.button("Analyze Resume with AI"):

    model_gemini = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""
    Analyze this resume.

    Give:
    1. Strengths
    2. Weaknesses
    3. Missing Skills
    4. Career Suggestions

    Resume:
    {text}
    """

    response = model_gemini.generate_content(prompt)

    st.subheader("AI Resume Analysis")

    st.write(response.text)

    # Career Guidance

st.subheader("Career Guidance")

if predicted_role[0] == "AI Engineer":
    st.write("""
    Recommended Skills:
    • Deep Learning
    • TensorFlow
    • PyTorch
    • NLP
    • Generative AI

    Recommended Certifications:
    • Google AI
    • AWS ML Specialty
    • IBM AI Engineering
    """)

elif predicted_role[0] == "Data Scientist":
    st.write("""
    Recommended Skills:
    • Python
    • Machine Learning
    • Statistics
    • SQL
    • Data Visualization
    """)

elif predicted_role[0] == "Software Engineer":
    st.write("""
    Recommended Skills:
    • DSA
    • Java
    • Python
    • OOP
    • System Design
    """)

    # Interview Questions

st.subheader("Interview Questions")

prompt_questions = f"""
Generate 10 interview questions for {predicted_role[0]}
"""
import google.generativeai as genai
import streamlit as st

genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

model_gemini = genai.GenerativeModel("gemini-2.5-flash")

questions = model_gemini.generate_content(prompt_questions)

st.write(questions.text)

st.subheader("Interview Questions")

questions = [
    "Tell me about yourself.",
    "Explain your final year project.",
    "What is Machine Learning?",
    "What are your strengths?",
    "What are your weaknesses?",
    "Why should we hire you?",
    "What is Random Forest?",
    "Explain Python OOP concepts.",
    "Describe a challenging project.",
    "Where do you see yourself in 5 years?"
]

for q in questions:
    st.write("•", q)


st.subheader("Resume Improvement Tips")

if resume_score < 90:
    st.warning("Add more projects and certifications")
else:
    st.success("Excellent Resume!")
