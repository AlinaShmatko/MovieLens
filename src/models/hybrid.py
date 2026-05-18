import numpy as np


class HybridRecommender:

    def __init__(
        self,
        svd_model,
        content_model,
        alpha=0.8,
        similarity_threshold=0.3,
        min_history=5
    ):

        self.svd = svd_model
        self.content = content_model

        self.alpha = alpha
        self.similarity_threshold = similarity_threshold
        self.min_history = min_history

        self.user_history = {}

    def fit_history(self, train_df):

        self.user_history = (
            train_df
            .groupby("userId")["movieId"]
            .apply(list)
            .to_dict()
        )

    def predict(self, user_ids, movie_ids):

        preds = []

        for user_id, movie_id in zip(user_ids, movie_ids):

            # 1. SVD prediction
            svd_pred = self.svd.predict([user_id],[movie_id])[0]

            # 2. cold-start fallback
            if user_id not in self.user_history:
                preds.append(svd_pred)
                continue

            if movie_id not in self.content.indices:
                preds.append(svd_pred)
                continue

            watched_movies = self.user_history[user_id]

            # insufficient history
            if len(watched_movies) < self.min_history:
                preds.append(svd_pred)
                continue

            # 3. content similarities
            idx = self.content.indices[movie_id]

            sims = []

            for watched in watched_movies:

                if watched not in self.content.indices:
                    continue

                watched_idx = self.content.indices[watched]
                sim = self.content.sim_matrix[idx][watched_idx]

                # use only strong similarities
                if sim >= self.similarity_threshold:
                    sims.append(sim)

            # 4. weak content signal
            if len(sims) == 0:
                preds.append(svd_pred)
                continue

            # 5. content score
            content_signal = np.mean(sims)

            # normalize to rating scale
            content_score = 1 + content_signal * 4

            # 6. confidence-aware weighting
            confidence = min(len(sims) / 10, 1.0)

            adaptive_alpha = self.alpha + (1 - confidence) * 0.15
            adaptive_alpha = min(adaptive_alpha, 0.95)

            final = (
                adaptive_alpha * svd_pred
                + (1 - adaptive_alpha) * content_score
            )

            final = np.clip(final, 0.5, 5.0)

            preds.append(final)

        return preds