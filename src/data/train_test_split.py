
def time_split(ratings, split_ratio=0.8):
    ratings = ratings.sort_values("timestamp")
    split_index = int(len(ratings) * split_ratio)

    train = ratings.iloc[:split_index]
    test = ratings.iloc[split_index:]

    return train, test