import numpy as np
import pandas as pd

#Matrix Factorization using Stochastic Gradient Descent (SGD)
class MatrixFactorization:

    def __init__(self, n_factors=10, learning_rate=0.01, reg=0.02, n_iters=50):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg = reg
        self.n_iters = n_iters

    def fit(self, ratings_df: pd.DataFrame):
        users = ratings_df['user_id'].unique()
        items = ratings_df['item_id'].unique()

        self.user_to_index = {u: i for i, u in enumerate(users)}
        self.item_to_index = {i: j for j, i in enumerate(items)}

        n_users = len(users)
        n_items = len(items)

        self.P = np.random.normal(scale=1./self.n_factors, size=(n_users, self.n_factors))
        self.Q = np.random.normal(scale=1./self.n_factors, size=(n_items, self.n_factors))

        for _ in range(self.n_iters):
            for _, row in ratings_df.iterrows():
                u = self.user_to_index[row['user_id']]
                i = self.item_to_index[row['item_id']]
                r = row['rating']

                pred = np.dot(self.P[u], self.Q[i])
                err = r - pred

                self.P[u] += self.learning_rate * (err * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.learning_rate * (err * self.P[u] - self.reg * self.Q[i])

    def predict(self, user_id, item_id):
        if user_id not in self.user_to_index or item_id not in self.item_to_index:
            return 0

        u = self.user_to_index[user_id]
        i = self.item_to_index[item_id]

        return np.dot(self.P[u], self.Q[i])
