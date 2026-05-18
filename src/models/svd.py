from surprise import SVD, Dataset, Reader
import os
import pickle

class SVDRecommender_old:
    def __init__(self, n_factors=100, lr_all=0.005, reg_all=0.02):
        self.model = SVD(
            n_factors=n_factors,
            lr_all=lr_all,
            reg_all=reg_all
        )
        self.reader = Reader(rating_scale=(0.5, 5.0))

    def fit(self, train_df):
        data = Dataset.load_from_df(
            train_df[["userId", "movieId", "rating"]],
            self.reader
        )

        trainset = data.build_full_trainset()
        self.model.fit(trainset)

    def predict(self, user_id, movie_id):
        return self.model.predict(user_id, movie_id).est



class SVDRecommender:
    def __init__(self, n_factors=100, lr_all=0.005, reg_all=0.02,
                 model_path="models/svd.pkl"):

        self.model_path = model_path
        self.model = None

        self.params = {
            "n_factors": n_factors,
            "lr_all": lr_all,
            "reg_all": reg_all
        }

        self.reader = Reader(rating_scale=(0.5, 5.0))

    def fit(self, train_df):

        # 👉 1. якщо модель вже є — просто завантажуємо
        if os.path.exists(self.model_path):
            print("Loading saved SVD model...")
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            return

        # 👉 2. інакше тренуємо
        print("Training SVD model...")

        data = Dataset.load_from_df(
            train_df[["userId", "movieId", "rating"]],
            self.reader
        )

        trainset = data.build_full_trainset()

        self.model = SVD(**self.params)
        self.model.fit(trainset)

        # 👉 3. зберігаємо
        os.makedirs("models", exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)

        print("Model saved to", self.model_path)

    # def predict(self, user_id, movie_id):
    #     return self.model.predict(user_id, movie_id).est
    def predict(self, user_ids, movie_ids):
        preds = []
        for u, i in zip(user_ids, movie_ids):
            preds.append(self.model.predict(u, i).est)
        return preds