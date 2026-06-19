import numpy as np 
from numpy.typing import NDArray
from numpy.typing import ArrayLike
from collections.abc import Callable
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph

class SpectralClustering:
    def __init__(self, k: int, s: Callable[[ArrayLike, ArrayLike], float]):
        self.k = k
        self.s = s

        self.kmeans = KMeans(self.k)

    def fit_predict(self, X) -> NDArray:
        # Step 1: Construct similarity graph and compute its Laplacian

        L = self.calculate_laplacian(X)

        # Step 2: Compute the eigenvectors of the Laplacian L and calculate V

        _, eigvec = np.linalg.eig(L)
        V = eigvec[:, :self.k]

        # Step 3: Use KMeans on V and return results
        return self.kmeans.fit_predict(V)
    
    def calculate_laplacian(self, X) -> NDArray:
        n = X.shape[0]

        A = np.zeros((n,n))
        D = np.zeros((n,n))

        for i in range(n):
            for j in range(n):
                A[i, j] = self.s(X[i], X[j])

        for i in range(n):
            D[i, i] = np.sum([A[i, j] for j in range(n)])

        return D-A

    def fit_predict_kneighbors(self, X, n_neighbors) -> NDArray:
        # Fit predict with a KNN-graph embedding
        n = X.shape[0]

        # Construct the similarity graph from the K-nearest neighbours
        G = kneighbors_graph(X, n_neighbors, mode="connectivity", include_self=True)
        G = G.toarray() # type: ignore

        # Make the matrix symmetric by connecting only mutual nearest neighbours
        for i in range(n):
            for j in range(n):
                if not (G[i,j] == 1 and G[j,i] == 1):
                    G[i,j] = 0
                    G[j,i] = 0

        D = np.zeros((n,n))
        
        for i in range(n):
            D[i, i] = np.sum([G[i, j] for j in range(n)])

        L = D-G

        _, eigvec = np.linalg.eig(L)
        V = eigvec[:, :self.k]

        return self.kmeans.fit_predict(V)
