import numpy as np 
from typing import List
from numpy.linalg.linalg import SVDResult
from numpy.typing import ArrayLike, NDArray
from collections.abc import Callable
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph

class SVDSpectralClustering:
    def __init__(self, k: int, l: int, s: Callable[[ArrayLike, ArrayLike], float]):
        self.k = k
        self.s = s
        self.l = l

        self.kmeans = KMeans(self.k)

    def fit_predict(self, X) -> NDArray:
        # Step 1: Construct similarity graph and get its SVD
        U, S, Vh = self.get_similarity_svd(X)
        U_ = U[:, :self.l]

        # Step 2: Use KMeans on U_ and return output
        return self.kmeans.fit_predict(U_)
    
    def get_similarity_svd(self, X) -> SVDResult:
        # Can be used to deduce information about the data, to deduct the parameters of the algorithm
        n = X.shape[0]

        A = np.zeros((n,n))

        for i in range(n):
            for j in range(n):
                A[i, j] = self.s(X[i], X[j])

        return np.linalg.svd(A)

    def fit_predict_graph(self, G) -> NDArray:
        # In case the adjacency graph has been already given
        U, S, Vh = np.linalg.svd(G)
        U_ = U[:, :self.l]

        return self.kmeans.fit_predict(U_)
    
    def fit_predict_kneighbors(self, X, n_neighbors) -> NDArray:
        # Construct the similarity graph from the K-nearest neighbours
        G = kneighbors_graph(X, n_neighbors, mode="connectivity", include_self=True)
        G = G.toarray() # type: ignore

        return self.fit_predict_graph(G)

