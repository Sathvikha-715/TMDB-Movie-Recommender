import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================= PAGE CONFIG =================
st.set_page_config(page_title="AI Movie Recommender", page_icon="🎬", layout="centered")

st.title("🎬 AI Movie Recommendation System")
st.write("Content-Based Movie Recommender using TMDB Dataset")

# ================= LOAD & PROCESS DATA =================
@st.cache_data
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")

    movies = movies.merge(credits, on="title")

    movies = movies[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]

    # Convert stringified lists to real lists
    def convert(obj):
        return [i["name"] for i in ast.literal_eval(obj)]

    def convert_cast(obj):
        L = []
        for i in ast.literal_eval(obj)[:3]:
            L.append(i["name"])
        return L

    def fetch_director(obj):
        for i in ast.literal_eval(obj):
            if i["job"] == "Director":
                return [i["name"]]
        return []

    movies["genres"] = movies["genres"].apply(convert)
    movies["keywords"] = movies["keywords"].apply(convert)
    movies["cast"] = movies["cast"].apply(convert_cast)
    movies["crew"] = movies["crew"].apply(fetch_director)

    movies["overview"] = movies["overview"].fillna("")

    # Combine all features
    movies["tags"] = (
        movies["overview"] + " " +
        movies["genres"].apply(lambda x: " ".join(x)) + " " +
        movies["keywords"].apply(lambda x: " ".join(x)) + " " +
        movies["cast"].apply(lambda x: " ".join(x)) + " " +
        movies["crew"].apply(lambda x: " ".join(x))
    )

    new_df = movies[["movie_id", "title", "tags"]]
    new_df["tags"] = new_df["tags"].apply(lambda x: x.lower())

    return new_df

movies = load_data()

# ================= BUILD SIMILARITY =================
@st.cache_data
def build_similarity(data):
    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(data["tags"]).toarray()
    similarity = cosine_similarity(vectors)
    return similarity

similarity = build_similarity(movies)

# ================= RECOMMENDER =================
def recommend(movie):
    if movie not in movies["title"].values:
        return []

    index = movies[movies["title"] == movie].index[0]
    distances = list(enumerate(similarity[index]))
    distances = sorted(distances, key=lambda x: x[1], reverse=True)

    recommended = []
    for i in distances[1:6]:
        recommended.append(movies.iloc[i[0]].title)

    return recommended

# ================= UI =================
movie_list = sorted(movies["title"].unique())
selected_movie = st.selectbox("Choose a movie:", movie_list)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)

    st.subheader("Recommended Movies:")
    for movie in recommendations:
        st.write("⭐", movie)

st.divider()
st.caption("Built by Sathvikha Reddy • Movie Recommendation System")
