# -*- coding: utf-8 -*-

from utils import *
from walks import multi_simulate_walks, single_simulate_walks
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from embedding.matrix_factorization import MatrixFactorization
from scipy import sparse

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from model import structing

data_folder = '../graph/'
attribute_folder = '../attribute/'
alignment_folder = '../alignment/'


def CENALP(G1, G2, q, attr1, attr2, attribute, alignment_dict, alignment_dict_reversed,
           layer, align_train_prop, alpha, c, multi_walk, neg_sampling=False, embedding_method_class = '', embedding_method_kind = ''):
    iteration = 1
    anchor = 0
    mul = int(np.max([np.max(G1.nodes()), np.max(G2.nodes())]))

    if len(attribute) != 0:
        attribute = attribute.T
        attribute = pd.DataFrame(attribute, index=list(G1.nodes()),
                                 columns=list(G2.nodes()))
    if align_train_prop == 0:
        seed_list1 = []
        seed_list2 = []
    else:
        seed_list1 = list(
            np.random.choice(list(alignment_dict.keys()), int(align_train_prop * len(alignment_dict)), replace=False))
        seed_list2 = [alignment_dict[seed_list1[x]] for x in range(len(seed_list1))]
    seed_l1 = seed_list1.copy()
    seed_l2 = seed_list2.copy()
    G1_degree_dict = cal_degree_dict(list(G1.nodes()), G1, layer)
    G2_degree_dict = cal_degree_dict(list(G2.nodes()), G2, layer)
    seed_list_num = len(seed_list1)

    k = seed_link(seed_list1, seed_list2, G1, G2, anchor=anchor)
    print()
    k = np.inf
    test_edges_final1 = np.array(list(set(G1.edges()) - set(
        [(alignment_dict_reversed.get(edge[0], 0), alignment_dict_reversed.get(edge[1], 0)) for edge in G2.edges()])))
    test_edges_final2 = np.array(list(set(G2.edges()) - set(
        [(alignment_dict.get(edge[0], 0), alignment_dict.get(edge[1], 0)) for edge in G1.edges()])))

    pred_list1, pred_list2 = [], []

    while True:
        print('------ The current iteration : {} ------'.format(iteration))
        iteration += 1
        index = list(G1.nodes())
        columns = list(G2.nodes())

        index = list(set(index) - set(seed_list1))
        columns = list(set(columns) - set(seed_list2))

        columns = [x + mul + 1 for x in columns]

        if k != 0:
            print('structing...', end='')
            G1_degree_dict = cal_degree_dict(list(G1.nodes()), G1, layer)
            G2_degree_dict = cal_degree_dict(list(G2.nodes()), G2, layer)
            struc_neighbor1, struc_neighbor2, struc_neighbor_sim1, struc_neighbor_sim2 = \
                structing(layer, G1, G2, G1_degree_dict, G2_degree_dict, attribute, alpha, c)
            print('finished!')

        G1_nodes_numb = len(G1.nodes())
        G2_nodes_numb = len(G2.nodes())

        big_graph = np.zeros((G1_nodes_numb + G2_nodes_numb, G1_nodes_numb + G2_nodes_numb))
        G1_dict = {}
        for i in range(len(list(G1.nodes()))):
            G1_dict[list(G1.nodes())[i]] = i
        G2_dict = {}
        for i in range(len(list(G2.nodes()))):
            G2_dict[list(G2.nodes())[i]] = i

        for edge in G1.edges():
            big_graph[G1_dict[edge[0]]][G1_dict[edge[1]]] = 1
            big_graph[G1_dict[edge[1]]][G1_dict[edge[0]]] = 1

        for edge in G2.edges():
            big_graph[G2_dict[edge[0]] + G1_nodes_numb][G2_dict[edge[1]] + G1_nodes_numb] = 1
            big_graph[G2_dict[edge[1]] + G1_nodes_numb][G2_dict[edge[0]] + G1_nodes_numb] = 1

        for i in range(len(seed_list1)):
            big_graph[G1_dict[seed_list1[i]]][G2_dict[seed_list2[i]] + G1_nodes_numb] = 1
            big_graph[G2_dict[seed_list2[i]] + G1_nodes_numb][G1_dict[seed_list1[i]]] = 1

        for n1 in struc_neighbor1:
            for n2 in struc_neighbor1[n1]:
                big_graph[G1_dict[n1]][G2_dict[n2] + G1_nodes_numb] = 1
                big_graph[G2_dict[n2] + G1_nodes_numb][G1_dict[n1]] = 1

        for n2 in struc_neighbor2:
            for n1 in struc_neighbor2[n2]:
                big_graph[G1_dict[n1]][G2_dict[n2] + G1_nodes_numb] = 1
                big_graph[G2_dict[n2] + G1_nodes_numb][G1_dict[n1]] = 1

        print('big_graph.sum()')
        print(big_graph.sum())

        model_01 = MatrixFactorization(embedding_dimension=64, similarity_measure='sum_power_tran')

        s_big_graph = sparse.csr_matrix(big_graph)  # Here's the initialization of the sparse matrix.
        model_01.setup_model_input(s_big_graph)
        emb = model_01.learn_embedding(10000)

        # print('walking...', end='')
        # if multi_walk == True:
        #     multi_simulate_walks(G1, G2, q, struc_neighbor1, struc_neighbor2,
        #                          struc_neighbor_sim1, struc_neighbor_sim2,
        #                          seed_list1, seed_list2,
        #                          num_walks=20, walk_length=80, workers=20)
        # else:
        #     single_simulate_walks(G1, G2, q, struc_neighbor1, struc_neighbor2,
        #                           struc_neighbor_sim1, struc_neighbor_sim2,
        #                           seed_list1, seed_list2,
        #                           num_walks=20, walk_length=80, workers=20)
        # walks = LineSentence('random_walks.txt')
        # print('finished!')
        # print('embedding...', end='')
        # if neg_sampling:
        #     model = Word2Vec(walks, size=64, window=5, min_count=0, hs=0, negative=10, sg=1, workers=32, iter=5)
        # else:
        #     model = Word2Vec(walks, size=64, window=5, min_count=0, hs=1, sg=1, workers=32, iter=5)
        # print('finished!')
        if len(columns) == 0 or len(index) == 0:
            break
        if len(alignment_dict) == len(seed_list1):
            break
        columns = [x - mul - 1 for x in columns]

        # embedding1 = np.array([model.wv[str(x)] for x in index])
        # embedding2 = np.array([model.wv[str(x + mul + 1)] for x in columns])

        embedding1 = np.array([np.array(emb[list(G1.nodes()).index(x)]).reshape(64) for x in index])
        embedding2 = np.array([np.array(emb[list(G2.nodes()).index(x) + G1_nodes_numb]).reshape(64) for x in columns])

        cos = cosine_similarity(embedding1, embedding2)
        adj_matrix = np.zeros((len(index) * len(columns), 3))
        for i in range(len(index)):
            for j in range(len(columns)):
                adj_matrix[i * len(columns) + j, 0] = index[i]
                adj_matrix[i * len(columns) + j, 1] = columns[j]
                adj_matrix[i * len(columns) + j, 2] = cos[i, j]

        adj_matrix[:, 2] = list(map(clip, adj_matrix[:, 2]))
        if len(seed_list1) != 0:
            adj_matrix2 = caculate_jaccard_coefficient(G1, G2, seed_list1, seed_list2, index, columns)
            adj_matrix[:, 2] *= adj_matrix2[:, 2]

        adj_matrix = adj_matrix[np.argsort(-adj_matrix[:, 2])]

        seed1 = []
        seed2 = []
        # np.mean(np.array(seed1)==np.array(seed2))
        len_adj_matrix = len(adj_matrix)
        if len_adj_matrix != 0:

            len_adj_matrix = len(adj_matrix)
            # T = np.max([int(len(alignment_dict) / 100), len(seed_list1)/2])
            T = np.max([5, int(len(alignment_dict) / 100 * (1.5 ** (iteration - 1)))])
            # if len(index_neighbors) == 0:
            #    T = int(len(alignment_dict) / 100)
            # else:
            #    T = max([int(len(alignment_dict) / 100), int(len(index_neighbors)/2)])
            # T = len(alignment_dict) / 10
            while len(adj_matrix) > 0 and T > 0:
                T -= 1
                node1, node2 = int(adj_matrix[0, 0]), int(adj_matrix[0, 1])
                seed1.append(node1)
                seed2.append(node2)
                adj_matrix = adj_matrix[adj_matrix[:, 0] != node1, :]
                adj_matrix = adj_matrix[adj_matrix[:, 1] != node2, :]
            anchor = len(seed_list1)

        anchor = len(seed_list1)
        seed_list1 += seed1
        seed_list2 += seed2
        print('Add seed nodes : {}'.format(len(seed1)), end='\t')

        count = 0
        for i in range(len(seed_list1)):
            try:
                if alignment_dict[seed_list1[i]] == seed_list2[i]:
                    count += 1
            except:
                continue
        print('All seed accuracy : %.2f%%' % (100 * count / len(seed_list1)))

        count -= seed_list_num
        precision = 100 * count / (len(seed_list1) - seed_list_num)
        recall = 100 * count / (len(alignment_dict) - seed_list_num)

        pred1, pred2 = seed_link_lr(emb, G1, G2, seed_list1, seed_list2,
                                    mul, test_edges_final1, test_edges_final2, alignment_dict, alignment_dict_reversed)

        G1.add_edges_from(pred1)
        G2.add_edges_from(pred2)
        print('Add seed links: {}'.format(len(pred1) + len(pred2)))

        # pred_list1 += list([[alignment_dict[edge[0]], alignment_dict[edge[1]]] for edge in pred1])
        # pred_list2 += list([[alignment_dict_reversed[edge[0]], alignment_dict_reversed[edge[1]]] for edge in pred2])
        for edge in pred1:
            if alignment_dict.get(edge[0], -1) != -1 and alignment_dict.get(edge[1], -1) != -1:
                pred_list1 += [[alignment_dict[edge[0]], alignment_dict[edge[1]]]]
        for edge in pred2:
            if alignment_dict_reversed.get(edge[0], -1) != -1 and alignment_dict_reversed.get(edge[1], -1) != -1:
                pred_list2 += [[alignment_dict_reversed[edge[0]], alignment_dict_reversed[edge[1]]]]

        sub1 = np.sum([G1.has_edge(edge[0], edge[1]) for edge in pred_list2])
        sub2 = np.sum([G2.has_edge(edge[0], edge[1]) for edge in pred_list1])
        if (len(pred_list2) + len(pred_list1)) == 0:
            precision2 = 0
        else:
            precision2 = (sub1 + sub2) / (len(pred_list2) + len(pred_list1)) * 100
        recall2 = (sub1 + sub2) / (len(test_edges_final1) + len(test_edges_final2)) * 100
        print('Precision : %.2f%%\tRecall :  %.2f%%' % (precision, recall))
        print('Link Precision:: %.2f%%\tRecall :  %.2f%%' % (precision2, recall2))

        # evaluate_lp(G1, G2, emb, mul)

    # embedding1 = np.array([model.wv[str(x)] for x in list(G1.nodes())])
    # embedding2 = np.array([model.wv[str(x + mul + 1)] for x in list(G2.nodes())])
    # adj = np.array(caculate_jaccard_coefficient(G1, G2, seed_list1, seed_list2, list(G1.nodes()), list(G2.nodes())))
    # L = [[adj[i * G1.number_of_nodes() + j, 2] for j in range(G2.number_of_nodes())] for i in range(G1.number_of_nodes())]
    S = cosine_similarity(embedding2, embedding1)
    return S, precision, seed_l1, seed_l2
