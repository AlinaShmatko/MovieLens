## 🎬 MovieLens Recommender System 

A production-style movie recommendation system built on the MovieLens 20M dataset.   
This project explores multiple recommendation approaches:   
Baseline recommendation models    
Matrix Factorization (SVD)    
Content-Based Filtering   
Hybrid Recommender Systems   
The goal of the project was not only to improve recommendation quality, but also to learn how to structure a real-world machine learning project using modular architecture and reproducible evaluation pipelines.  

---
##  📌 Dataset   
Dataset: MovieLens 20M Dataset   
The dataset contains:   
- 20 million movie ratings   
- 138k users   
- 26k movies   
- movie metadata (genres, tags)    

Main files used:
- ratings.csv
- movies.csv
- tags.csv
----
##  🚀 Project Goals
The project focuses on:   
- Building collaborative filtering models from scratch   
- Comparing recommendation approaches using RMSE   
- Understanding recommender system pipelines   
- Implementing modular ML architecture   
- Combining collaborative and content-based methods 
- Exploring hybrid recommendation strategies
---
## ⚙️ Requirements

- Python 3.10  
  (Required due to compatibility constraints with the `scikit-surprise` library, which relies on compiled extensions that are not fully supported on Python 3.11+ and 3.12+.)
---
## 🧠 Implemented Models
1. Baseline Models       
Implemented:
- Global Mean
- User Bias
- Item Bias
- User + Item Bias   
These models establish a baseline before applying more advanced matrix factorization methods.
2. Matrix Factorization — SVD
Implemented using Surprise.
The model learns latent user and item embeddings:
$$$
r_{u,i}=μ+b_u+b_i + p^T_uq_i
$$$
Hyperparameter tuning was performed using GridSearchCV.
The trained model is serialized and stored in:   
```
models/svd.pkl
```
3. Content-Based Filtering
Implemented using:
- TF-IDF vectorization
- cosine similarity    
Movie genres were cleaned and transformed into feature vectors.
#### Genre preprocessing
```
valid_movies["genres"] = valid_movies["genres"].str.replace("|", " ", regex=False)
```
Movies without genres were removed:
```
valid_mask = self.movies["genres"] != "(no genres listed)"
```
This improved genre similarity quality and reduced noise in the recommendation pipeline.
4. Hybrid Recommender System
The hybrid recommender combines:
- Collaborative Filtering (SVD)
- Content-Based similarity
Hybrid prediction:
$$$
Prediction=α⋅SVD+(1−α)⋅ContentSimilarity
$$$
This approach improves robustness and partially mitigates cold-start problems.
---
## 📊 Evaluation
Evaluation metric:   
- RMSE (Root Mean Squared Error)  

Time-based train/test split was used to prevent data leakage and better simulate real-world recommendation scenarios.

---
## 📈 Results
| Model            | RMSE   |
| ---------------- |--------|
| Global Mean      | 1.0200 |
| User Bias        | 1.0132 |
| Item Bias        | 0.9475 |
| User + Item Bias | 0.9422 |
| SVD              | 0.9282 |
| Hybrid            | 0.9304 |
#### Key Insights
- Bias-based models improved over the global mean baseline.
- Combining user and item biases further reduced RMSE.
- SVD achieved the best performance (RMSE = 0.9282).
- Confidence-aware hybrid logic substantially improved hybrid stability
- Content-based similarity improved semantic coherence of recommendations even when RMSE gains were limited.
- Better recommendation quality does not always correspond to lower RMSE.
---
## 🏗️ Project Architecture
The project follows a modular ML engineering structure:
- src/data/ → data loading and preprocessing
- src/models/ → recommender models
- src/evaluation/ → evaluation metrics
- notebooks/ → experimentation and analysis
- models/ → serialized trained models
---

```text
MovieLens/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── models/
│   ├── svd.pkl               
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_svd.ipynb
│   ├── 03_model_comparison.ipynb
│   ├── 04_hybrid_model.ipynb
│
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   ├── preprocessing.py
│   │   ├── train_test_split.py
│   │
│   ├── models/
│   │   ├── baseline.py
│   │   ├── content_based.py
│   │   ├── hybrid.py
│   │   ├── svd.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │
│   ├── config.py
│
├── main.py
├── requirements.txt
├── README.md
```
---
## ⚙️ Installation
Install dependencies:
```
pip install -r requirements.txt
```
### Dataset Setup:
## 📥 Dataset Setup

This project uses the MovieLens 20M dataset.
Download the dataset from:

https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset   
After downloading, place the files into:
data/raw/

Required files:
- rating.csv
- movie.csv
- tag.csv
```
python main.py
```
---
## 🔮 Future Improvements
Potential future improvements:
- implicit feedback modeling
- deep learning recommenders
- ANN similarity search
- deployment with FastAPI
- real-time recommendations
- user embedding visualization
----
