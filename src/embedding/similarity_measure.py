'''
File to produce following similarity measures:
- Adjacency Matrix A
- Laplacian P=D^−1A
- Deep Walk / Symmetric, Normalized Laplacian D^(−1/2)AD^(−1/2)
- NetMF
- Personalized Page Rank
- Sum of Power of Transitions
Input: Adjacency Matrix of a Graph (Dataset), NxN Matrix
Initialize: None
Output: Similarity Measure, NxN Matrix
'''

import sys

import numpy

sys.path.insert(0, '../')


# helper:
def compute_degree_matrix(adj_np):
    degree = adj_np.sum(axis=1)
    return degree


# Transition P=D^−1A
def compute_transition(adj_np):
    adj_gpu = adj_np.toarray()
    degree = compute_degree_matrix(adj_np)
    inv_degree = numpy.diagflat(1 / degree)
    P = numpy.matmul(inv_degree,adj_gpu)
    return P


def compute_sum_power_tran(adj_np, T=10):
    # P=D^−1A
    matrix = compute_transition(adj_np)
    P = numpy.zeros_like(matrix)
    for i in range(0, T + 1):
        P = P + numpy.power(matrix,i)
    return P
