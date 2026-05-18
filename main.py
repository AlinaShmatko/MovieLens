import numpy as np
from sklearn.metrics import mean_squared_error
from src.data.load_data import load_ratings, load_movies
from src.data.train_test_split import time_split
from src.data.preprocessing import convert_timestamp
from src.models.baseline import (GlobalMeanModel,UserBiasModel,ItemBiasModel,UserItemBiasModel)
from src.models.svd import SVDRecommender
from src.models.content_based import ContentBasedRecommender, ContentBasedRecommender_GT
from src.models.hybrid import HybridRecommender


# EVALUATION FUNCTION (BASE MODELS)
def evaluate_model(model, train, test):
    model.fit(train)
    preds = model.predict(test["userId"].values,
                          test["movieId"].values)
    rmse = np.sqrt(mean_squared_error(test["rating"], preds))
    return rmse


# EVALUATION FUNCTION (HYBRID)
def evaluate_hybrid(model, test):
    preds = model.predict(test["userId"].values,
                          test["movieId"].values)
    rmse = np.sqrt(mean_squared_error(test["rating"], preds))
    return rmse


def main():
    # 1. Load data
    ratings = load_ratings()
    movies = load_movies()

    # 2. Preprocess
    ratings = convert_timestamp(ratings)

    # 3. Time-based split
    train, test = time_split(ratings)

    print("Train size:", len(train))
    print("Test size:", len(test))

    # BASELINE MODELS
    print("\n===== BASELINES =====")

    print("Global Mean:",
          evaluate_model(GlobalMeanModel(), train, test))

    print("User Bias:",
          evaluate_model(UserBiasModel(), train, test))

    print("Item Bias:",
          evaluate_model(ItemBiasModel(), train, test))

    print("User + Item Bias:",
          evaluate_model(UserItemBiasModel(), train, test))

    # SVD
    print("\n===== SVD =====")

    svd = SVDRecommender(
        n_factors=100,
        lr_all=0.005,
        reg_all=0.02)

    rmse = evaluate_model(svd, train, test)
    print("SVD RMSE:", rmse)

    # CONTENT MODEL
    print("\n===== CONTENT genres =====")

    content = ContentBasedRecommender(movies)
    content.fit()

    print("\n===== CONTENT genres + tags =====")

    content_gt = ContentBasedRecommender_GT(movies)
    content_gt.fit()

    # HYBRID EXAMPLE
    print("\n===== HYBRID =====")

    hybrid = HybridRecommender(svd, content, alpha=0.7)
    hybrid.fit_history(train)
    hybrid_rmse = evaluate_hybrid(hybrid, test)
    print("Hybrid RMSE:", hybrid_rmse)


if __name__ == "__main__":
    main()