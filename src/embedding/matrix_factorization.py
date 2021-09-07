import os
import sys

import numpy
from numpy import linalg

from .similarity_measure import compute_sum_power_tran

sys.path.append('./')
sys.path.append(os.path.realpath(__file__))

from .static_graph_embedding import StaticGraphEmbedding
from .cpu_hosvd import cpu_tsvd


class MatrixFactorization(StaticGraphEmbedding):

    def __init__(self, embedding_dimension=64, similarity_measure="adjacency"):
        '''
        Parameters
        ----------
        embedding_dimension
            Number of elements in the embedding vector representing a node, default 64.
        sim_similarity_measure
            One of {'adjacency','laplacian', 'sym_normalized_laplacian', 'transition', 'NetMF', 'ppr', 'sum_power_tran'}, default 'adjacency'.
        '''
        self._embedding_dim = embedding_dimension
        self._similarity_measure = similarity_measure
        self._method_name = "Matrix_Fatorization"
        self._setup_done = False  # model input is not setup yet

    def setup_model_input(self, adj_mat, similarity_measure=None):
        '''
        Parameters
        ----------
        adj_mat
            Adjacency matrix of the dataset to be tested in numpy sparse format.
        sim_similarity_measure
            One of {'adjacency','laplacian', 'sym_normalized_laplacian', 'transition', 'NetMF', 'ppr', 'sum_power_tran'}, default 'adjacency'.
            If None, use similarity_measure from init

        Return
        ------
        No explicit return value.
        But class variable self._Mat is assigned the similarity measure in the form of a torch tensor.
        '''

        if (similarity_measure):
            self._similarity_measure = similarity_measure

        # transform input matrix to correct similiarity measure
        if (self._similarity_measure == "sum_power_tran"):
            self._Mat = compute_sum_power_tran(adj_mat)

        self._setup_done = True

    def get_method_name(self):
        '''        
        Return
        ------
        Name of embedding method as a string.
        '''
        return self._method_name

    def get_method_summary(self):
        '''        
        Return
        ------
        Name of entire model description including the name of the embedding method, 
        the name of the used similarity measure and the number of embedding dimensions.
        '''
        return f'{self._method_name}_{self._similarity_measure}_{self._embedding_dim}'

    def reset_epoch(self):
        '''   
        This method resets start and end point of the training.   

        Return
        ------
        -
        '''
        self._epoch_begin = 0
        self._epoch_end = 0

    def set_summary_folder(self, path):
        '''   
        This method creates a tensorboard summary writter which is used to log metrics during training and evaluation.
           
        Return
        ------
        -
        '''
        self._summary_path = path

    def get_summary_writer(self):
        '''   
        Return
        ------
        Reference to tensorboard summary writer (private class variable).
        '''
        return self._writer

    def learn_embedding(self, num_epochs):
        '''   
        This method uses Singular Value Decomposition to transform the input matrix of N x N down to N x embedding dimensions.
        The paramter num_epochs exists only for compatibility purposes and can be ignored.
        Return
        ------
        Embedding matrix of N x embedding dimensions as numpy matrix on the CPU
        '''
        if self._setup_done == False:
            raise ValueError('Model input parameters not defined.')

        U, S, V_trancated = cpu_tsvd(self._Mat, self._embedding_dim)
        V_trancated = V_trancated.transpose()
        U, S, V = linalg.svd(self._Mat)

        V_trancated = V[:, :self._embedding_dim]
        self._emb = numpy.matmul(self._Mat, V_trancated)

        # Put the embedding back on the CPU
        emb_np = self._emb

        return emb_np
