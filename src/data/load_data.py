import pandas as pd
import os
from pathlib import Path


# визначаємо корінь проєкту
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw"

def load_ratings():
    return pd.read_csv(os.path.join(DATA_PATH, "rating.csv"))

def load_movies():
    return pd.read_csv(os.path.join(DATA_PATH, "movie.csv"))

def load_tags():
    return pd.read_csv(os.path.join(DATA_PATH, "tag.csv"))

def load_genome_scores():
    return pd.read_csv(os.path.join(DATA_PATH, "genome_scores.csv"))

def load_genome_tags():
    return pd.read_csv(os.path.join(DATA_PATH, "genome_tags.csv"))


