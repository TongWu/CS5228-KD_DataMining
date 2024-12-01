import numpy as np
from networkx import neighbors

from sklearn.metrics.pairwise import euclidean_distances


def get_noise_dbscan(X, eps=0.0, min_samples=0):
    core_point_indices = []
    noise_point_indices = []

    #########################################################################################
    ### Your code starts here ###############################################################

    ### 2.1 a) Identify the indices of all core points
    distances = euclidean_distances(X)

    for i in range(X.shape[0]):
        neighbors = np.where(distances[i] <= eps)[0]
        if len(neighbors) >= min_samples:
            core_point_indices.append(i)

    ### Your code ends here #################################################################
    #########################################################################################

    #########################################################################################
    ### Your code starts here ###############################################################

    ### 2.1 b) Identify the indices of all noise points ==> noise_point_indices

    tmp = np.zeros(len(X), dtype=bool)

    for i in core_point_indices:
        neighbors = distances[i] <= eps
        tmp = np.logical_or(tmp, neighbors)

    noise_point_indices = np.where(~tmp)[0]

    ### Your code ends here #################################################################
    #########################################################################################

    return core_point_indices, noise_point_indices
