import streamlit as st
import plotly.express as px

from utils import load_interns, load_jobs
from model import (
    analyze_skill_gap,
    train_model,
    predict_job_cluster
)

# Page Configuration
st.set_page_config(
    page_title="Skill Gap Analysis Tool",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Skill Gap Analysis Tool")
st.write("Analyze intern skills and identify gaps compared to industry requirements.")

# Load datasets
interns = load_interns("interns.csv")
jobs = load_jobs("jobs.csv")

# Train the Machine Learning model
vectorizer, kmeans = train_model(jobs)

# Sidebar
# Sidebar
st.sidebar.title("📊 Skill Gap Analyzer")

st.sidebar.markdown("""
### About
This application analyzes an intern's current skills and compares them with industry job requirements using **TF-IDF**, **K-Means Clustering**, and **Skill Gap Analysis**.

**Technologies Used**
- 🐍 Python
- 📊 Streamlit
- 🤖 Scikit-learn
- 📈 Plotly
- 📝 Pandas
""")

st.sidebar.divider()

st.sidebar.header("🔍 Select Options")

intern_name = st.sidebar.selectbox(
    "Select Intern",
    interns["Name"]
)

job_title = st.sidebar.selectbox(
    "Select Target Job",
    jobs["Job_Title"]
)

# Get selected data
intern = interns[interns["Name"] == intern_name].iloc[0]
job = jobs[jobs["Job_Title"] == job_title].iloc[0]

matched, missing, score = analyze_skill_gap(
    intern["Skills"],
    job["Required_Skills"]
)

cluster = predict_job_cluster(
    job["Required_Skills"],
    vectorizer,
    kmeans
)

# Display Results
st.subheader("📋 Analysis Result")

st.write(f"**Intern:** {intern['Name']}")
st.write(f"**Current Role:** {intern['Current_Role']}")
st.write(f"**Target Job:** {job['Job_Title']}")
st.write(f"**Job Cluster:** Cluster {cluster}")

st.subheader("📊 Dashboard")

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric(
        label="Match Score",
        value=f"{score}%"
    )

with metric2:
    st.metric(
        label="Matched Skills",
        value=len(matched)
    )

with metric3:
    st.metric(
        label="Missing Skills",
        value=len(missing)
    )

st.progress(score / 100)

col1, col2 = st.columns(2)

with col1:
    st.subheader("✅ Matching Skills")
    for skill in matched:
        st.success(skill.title())

with col2:
    st.subheader("❌ Missing Skills")
    for skill in missing:
        st.error(skill.title())

st.subheader("📊 Skills Overview")

chart_data = {
    "Category": ["Matched", "Missing"],
    "Count": [len(matched), len(missing)]
}

fig = px.pie(
    chart_data,
    names="Category",
    values="Count",
    title="Skill Distribution",
    hole=0.45
)

st.plotly_chart(fig, use_container_width=True)       

st.subheader("📚 Recommended Courses")

courses = job["Recommended_Courses"].split(";")

for course in courses:
    st.info(course.strip())

st.divider()

st.markdown(
    """
    <div style='text-align:center; color:gray;'>
        <h4>Skill Gap Analysis Tool</h4>
        <p>AI & NLP Internship Project | Built with Streamlit, Scikit-learn & Plotly</p>
    </div>
    """,
    unsafe_allow_html=True
)    