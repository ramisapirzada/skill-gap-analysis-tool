from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


def train_model(job_data):
    """
    Train the TF-IDF Vectorizer and K-Means clustering model.
    """

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(job_data["Required_Skills"])

    kmeans = KMeans(
        n_clusters=5,
        random_state=42,
        n_init=10
    )

    kmeans.fit(tfidf_matrix)

    return vectorizer, kmeans


def predict_job_cluster(required_skills, vectorizer, kmeans):
    """
    Predict the cluster for a selected job.
    """

    vector = vectorizer.transform([required_skills])

    cluster = kmeans.predict(vector)[0]

    return cluster


def analyze_skill_gap(intern_skills, required_skills):
    """
    Compare intern skills with required job skills.
    """

    intern_set = {
        skill.strip().lower()
        for skill in intern_skills.split(",")
    }

    required_set = {
        skill.strip().lower()
        for skill in required_skills.split(",")
    }

    matched_skills = sorted(intern_set.intersection(required_set))

    missing_skills = sorted(required_set.difference(intern_set))

    match_score = round(
        (len(matched_skills) / len(required_set)) * 100,
        2
    )

    return matched_skills, missing_skills, match_score