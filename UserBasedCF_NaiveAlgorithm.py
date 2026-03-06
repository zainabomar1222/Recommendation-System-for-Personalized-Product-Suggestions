import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


#User-Based Collaborative Filtering using cosine similarity and k-nearest neighbors
class UserBasedCF:

    def __init__(self, k=5):
        self.k = k
        self.user_item_matrix = None
        self.similarity_matrix = None
        
        
#Build user-item matrix and compute user-user similarity
    def fit(self, ratings_df: pd.DataFrame):
        self.user_item_matrix = ratings_df.pivot_table(
            index='user_id', columns='item_id', values='rating'
        ).fillna(0)

        self.similarity_matrix = cosine_similarity(self.user_item_matrix)

        
#Predict rating for a given user-item pair
    def predict(self, user_id, item_id):
        if item_id not in self.user_item_matrix.columns:
            return 0

        if user_id not in self.user_item_matrix.index:
            return 0

        user_index = self.user_item_matrix.index.get_loc(user_id)

        sim_scores = self.similarity_matrix[user_index]
        item_ratings = self.user_item_matrix[item_id].values

        scores = list(zip(sim_scores, item_ratings))
        scores = [s for s in scores if s[1] != 0]

        scores = sorted(scores, key=lambda x: x[0], reverse=True)
        k_neighbors = scores[:self.k]

        if len(k_neighbors) == 0:
            return 0

        numerator = sum(sim * rating for sim, rating in k_neighbors)
        denominator = sum(abs(sim) for sim, _ in k_neighbors)

        return numerator / denominator if denominator != 0 else 0
