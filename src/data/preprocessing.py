import pandas as pd

def convert_timestamp(ratings):
    ratings["timestamp"] = pd.to_datetime(ratings["timestamp"])
    return ratings


def filter_users(ratings, min_ratings=20):
    user_counts = ratings.groupby("userId").size()
    active_users = user_counts[user_counts >= min_ratings].index
    return ratings[ratings["userId"].isin(active_users)]
