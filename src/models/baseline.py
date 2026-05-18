import numpy as np


class GlobalMeanModel:
    def fit(self, train):
        self.global_mean = train["rating"].mean()

    def predict(self, user_ids, movie_ids):
        return np.full(len(user_ids), self.global_mean)


class UserBiasModel:
    def fit(self, train):
        self.global_mean = train["rating"].mean()
        self.user_bias = train.groupby("userId")["rating"].mean() - self.global_mean

    def predict(self, user_ids, movie_ids):
        preds = []
        for u in user_ids:
            bu = self.user_bias.get(u, 0)
            preds.append(self.global_mean + bu)
        return preds


class ItemBiasModel:
    def fit(self, train):
        self.global_mean = train["rating"].mean()
        self.item_bias = train.groupby("movieId")["rating"].mean() - self.global_mean

    def predict(self, user_ids, movie_ids):
        preds = []
        for i in movie_ids:
            bi = self.item_bias.get(i, 0)
            preds.append(self.global_mean + bi)
        return preds


class UserItemBiasModel:
    def fit(self, train):
        self.global_mean = train["rating"].mean()
        self.user_bias = train.groupby("userId")["rating"].mean() - self.global_mean
        self.item_bias = train.groupby("movieId")["rating"].mean() - self.global_mean

    def predict(self, user_ids, movie_ids):
        preds = []
        for u, i in zip(user_ids, movie_ids):
            bu = self.user_bias.get(u, 0)
            bi = self.item_bias.get(i, 0)
            preds.append(self.global_mean + bu + bi)
        return preds