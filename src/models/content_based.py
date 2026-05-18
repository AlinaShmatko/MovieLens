from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    def __init__(self, movies):
        self.movies = movies
        self.indices = None
        self.sim_matrix = None


    def fit(self):
        # clean genres
        self.movies["genres"] = self.movies["genres"].fillna("")
        valid_mask = self.movies["genres"] != "(no genres listed)"
        valid_movies = self.movies[valid_mask].copy()
        valid_movies["genres"] = valid_movies["genres"].str.replace("|", " ", regex=False)

        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(valid_movies["genres"])

        self.sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        self.indices = {
            mid: idx for idx, mid in enumerate(valid_movies["movieId"])
        }

    def recommend(self, movie_id, top_n=10):
        idx = self.indices[movie_id]

        sim_scores = list(enumerate(self.sim_matrix[idx]))
        sim_scores = [x for x in sim_scores if x[0] != idx]
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        top = sim_scores[:top_n]
        movie_idx = [i[0] for i in top]

        return self.movies.iloc[movie_idx][["movieId", "title", "genres"]]


class ContentBasedRecommender_GT:
    def __init__(self, movies, tags=None):
        self.movies = movies.copy()
        self.tags = tags
        self.indices = None
        self.sim_matrix = None


    def fit(self):
        # 1. CLEAN GENRES
        self.movies["genres"] = self.movies["genres"].fillna("")
        self.movies["genres"] = self.movies["genres"].replace("(no genres listed)", "")
        self.movies["genres"] = self.movies["genres"].str.replace("|", " ", regex=False)

        # 2. PROCESS TAGS (NEW)
        if self.tags is not None:
            movie_tags = (
                self.tags.groupby("movieId")["tag"]
                .apply(lambda x: " ".join(x.astype(str)))
                .reset_index())
            movie_tags.columns = ["movieId", "user_tags"]

            self.movies = self.movies.merge(movie_tags, on="movieId", how="left")
        else:
            self.movies["user_tags"] = ""
        self.movies["user_tags"] = self.movies["user_tags"].fillna("")
        # 3. CREATE CONTENT TEXT
        self.movies["content"] = (
                self.movies["genres"] + " " + self.movies["user_tags"])

        # 4. TF-IDF
        tfidf = TfidfVectorizer(
            stop_words="english",
            max_features=5000)

        tfidf_matrix = tfidf.fit_transform(self.movies["content"])

        # 5. COSINE SIMILARITY

        self.sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        self.indices = {
            mid: idx for idx, mid in enumerate(self.movies["movieId"])
        }

    def recommend(self, movie_id, top_n=10):

        idx = self.indices[movie_id]

        sim_scores = list(enumerate(self.sim_matrix[idx]))
        sim_scores = [x for x in sim_scores if x[0] != idx]
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        top = sim_scores[:top_n]
        movie_idx = [i[0] for i in top]

        return self.movies.iloc[movie_idx][
            ["movieId", "title", "genres", "user_tags"]]
