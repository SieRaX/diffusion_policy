import warnings

import torch
import numpy as np
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors

def verify_sample_shapes(s1, s2, k):
    # Expects [N, D]
    assert len(s1.shape) == len(s2.shape) == 2
    # Check dimensionality of sample is identical
    assert s1.shape[1] == s2.shape[1]

def scipy_estimator(s1, s2, k=1):
    """KL-Divergence estimator using scipy's KDTree
    s1: (N_1,D) Sample drawn from distribution P
    s2: (N_2,D) Sample drawn from distribution Q
    k: Number of neighbours considered (default 1)
    return: estimated D(P|Q)
    """
    verify_sample_shapes(s1, s2, k)

    n, m = len(s1), len(s2)
    d = float(s1.shape[1])
    D = np.log(m / (n - 1))

    nu_d, nu_i = KDTree(s2).query(s1, k)
    rho_d, rhio_i = KDTree(s1).query(s1, k + 1)

    # KTree.query returns different shape in k==1 vs k > 1
    if k > 1:
        D += (d / n) * np.sum(np.log(nu_d[::, -1] / rho_d[::, -1]))
    else:
        D += (d / n) * np.sum(np.log(nu_d / rho_d[::, -1]))

    return D

def skl_efficient(s1, s2, k=1):
    """An efficient version of the scikit-learn estimator by @LoryPack
    s1: (N_1,D) Sample drawn from distribution P
    s2: (N_2,D) Sample drawn from distribution Q
    k: Number of neighbours considered (default 1)
    return: estimated D(P|Q)

    Contributed by Lorenzo Pacchiardi (@LoryPack)
    """
    verify_sample_shapes(s1, s2, k)

    n, m = len(s1), len(s2)
    d = float(s1.shape[1])

    s1_neighbourhood = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree").fit(s1)
    s2_neighbourhood = NearestNeighbors(n_neighbors=k, algorithm="kd_tree").fit(s2)

    s1_distances, indices = s1_neighbourhood.kneighbors(s1, k + 1)
    s2_distances, indices = s2_neighbourhood.kneighbors(s1, k)
    rho = s1_distances[:, -1]
    nu = s2_distances[:, -1]
    if np.any(rho == 0):
        warnings.warn(
            f"The distance between an element of the first dataset and its {k}-th NN in the same dataset "
            f"is 0; this causes divergences in the code, and it is due to elements which are repeated "
            f"{k + 1} times in the first dataset. Increasing the value of k usually solves this.",
            RuntimeWarning,
        )
    D = np.sum(np.log(nu / rho))

    return (d / n) * D + np.log(
        m / (n - 1)
    )  # this second term should be enough for it to be valid for m \neq n


def pytorch_kl_divergence(s1, s2, k=1, chunk_size=1000):
    """
    s1: (N_1,D) Sample drawn from distribution P
    s2: (N_2,D) Sample drawn from distribution Q
    k: Number of neighbours considered (default 1)
    chunk_size: Size of chunks to process at once to avoid memory issues
    """
    n, m = len(s1), len(s2)
    d = s1.shape[1]
    D = torch.tensor(np.log(m / (n - 1)), dtype=torch.float32, device=s1.device)
    
    # Process s2 distances in chunks
    s2_k_th_distances = []
    for i in range(0, n, chunk_size):
        chunk_end = min(i + chunk_size, n)
        s1_chunk = s1[i:chunk_end]
        
        # Compute distances for this chunk
        chunk_distances = []
        for j in range(0, m, chunk_size):
            s2_chunk = s2[j:min(j + chunk_size, m)]
            
            # Compute pairwise distances between chunks
            s1_expanded = s1_chunk.unsqueeze(1)  # [chunk_size, 1, D]
            s2_expanded = s2_chunk.unsqueeze(0)  # [1, chunk_size, D]
            dist = torch.norm(s1_expanded - s2_expanded, dim=2)  # [chunk_size, chunk_size]
            chunk_distances.append(dist)
            
        # Concatenate all distances for this s1 chunk
        chunk_distances = torch.cat(chunk_distances, dim=1)  # [chunk_size, m]
        # Get k-th smallest distance for each point in chunk
        chunk_k_th = torch.topk(chunk_distances, k=k, largest=False, dim=1).values.squeeze()
        s2_k_th_distances.append(chunk_k_th)
    
    s2_k_th_distance = torch.cat(s2_k_th_distances)
    
    # Process s1 distances in chunks
    s1_k_th_distances = []
    for i in range(0, n, chunk_size):
        chunk_end = min(i + chunk_size, n)
        s1_chunk = s1[i:chunk_end]
        
        # Compute distances for this chunk
        chunk_distances = []
        for j in range(0, n, chunk_size):
            s1_other = s1[j:min(j + chunk_size, n)]
            
            # Compute pairwise distances between chunks
            s1_expanded = s1_chunk.unsqueeze(1)
            s1_other_expanded = s1_other.unsqueeze(0)
            dist = torch.norm(s1_expanded - s1_other_expanded, dim=2)
            chunk_distances.append(dist)
            
        # Concatenate all distances for this chunk
        chunk_distances = torch.cat(chunk_distances, dim=1)  # [chunk_size, n]
        # Get (k+1)-th smallest distance for each point in chunk
        chunk_k_th = torch.topk(chunk_distances, k=k+1, largest=False, dim=1).values[:, -1].squeeze()
        s1_k_th_distances.append(chunk_k_th)
    
    s1_k_th_distance = torch.cat(s1_k_th_distances)
    
    # Compute final KL divergence
    D = D + (d/n)*torch.sum(torch.log(s2_k_th_distance/s1_k_th_distance))
    
    return D
