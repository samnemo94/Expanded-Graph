# -*- coding: utf-8 -*-
import random

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity


def loadG(data_folder, filename):
    G1 = nx.Graph()
    G2 = nx.Graph()
    G1_edges = pd.read_csv(data_folder + filename + '1.edges', names=['0', '1'])
    G1.add_edges_from(np.array(G1_edges))
    G2_edges = pd.read_csv(data_folder + filename + '2.edges', names=['0', '1'])
    G2.add_edges_from(np.array(G2_edges))
    return G1, G2


def loadG_link(data_folder, test_frac, filename):
    G1 = nx.Graph()
    G2 = nx.Graph()
    G1_edges = pd.read_csv(data_folder + filename + '1.edges', names=['0', '1'])
    G1.add_edges_from(np.array(G1_edges))
    G2_edges = pd.read_csv(data_folder + filename + '2_' + str(test_frac) + '.edges', names=['0', '1'])
    G2.add_edges_from(np.array(G2_edges))
    test_edges = pd.read_csv(data_folder + filename + '2_' + str(test_frac) + '_test.edges', names=['0', '1'])
    return G1, G2, test_edges


def load_attribute(attribute_folder, filename, G1, G2):
    G1_nodes = list(G1.nodes())
    G2_nodes = list(G2.nodes())
    attribute1 = pd.read_csv(attribute_folder + filename + 'attr1.csv', header=None, index_col=0)
    attribute2 = pd.read_csv(attribute_folder + filename + 'attr2.csv', header=None, index_col=0)
    attribute1 = np.array(attribute1.loc[G1_nodes, :])
    attribute2 = np.array(attribute2.loc[G2_nodes, :])
    attr_cos = cosine_similarity(attribute1, attribute2)
    # attr_cos = pd.DataFrame(attr_cos, index = attribute1.index, columns = attribute2.index)
    # attr_cos = attr_cos.loc[G1_nodes, G2_nodes]
    # attr_cos = np.array(attr_cos)
    return attr_cos, attribute1, attribute2


def greedy_match(X, G1, G2):
    G1_nodes = list(G1.nodes())
    G2_nodes = list(G2.nodes())
    m, n = X.shape
    x = np.array(flatten()).reshape(-1, )
    minSize = min(m, n)
    usedRows = np.zeros(n)
    usedCols = np.zeros(m)
    maxList = np.zeros(minSize)
    row = np.zeros(minSize)
    col = np.zeros(minSize)
    ix = np.argsort(-np.array(x))
    matched = 0
    index = 0
    while (matched < minSize):
        ipos = ix[index]
        jc = int(np.floor(ipos / n))
        ic = int(ipos - jc * n)
        if (usedRows[ic] != 1 and usedCols[jc] != 1):
            row[matched] = G1_nodes[ic]
            col[matched] = G2_nodes[jc]
            maxList[matched] = x[index]
            usedRows[ic] = 1
            usedCols[jc] = 1
            matched += 1
        index += 1;
    row = row.astype(int)
    col = col.astype(int)
    return zip(col, row)


def greedy_match_CENALP(X, G1_nodes, G2_nodes, minSize=10):
    m, n = X.shape
    x = np.array(flatten()).reshape(-1, )
    usedRows = np.zeros(n)
    usedCols = np.zeros(m)
    maxList = np.zeros(minSize)
    row = np.zeros(minSize)
    col = np.zeros(minSize)
    ix = np.argsort(-np.array(x))
    matched = 0
    index = 0
    while (matched < minSize):
        ipos = ix[index]
        jc = int(np.floor(ipos / n))
        ic = int(ipos - jc * n)
        if (usedRows[ic] != 1 and usedCols[jc] != 1):
            row[matched] = G1_nodes[ic]
            col[matched] = G2_nodes[jc]
            maxList[matched] = x[index]
            usedRows[ic] = 1
            usedCols[jc] = 1
            matched += 1
        index += 1;
    row = row.astype(int)
    col = col.astype(int)
    return zip(col, row)


def one2one_accuracy_supervised(S, G1, G2, alignment, seed_list1, seed_list2):
    ss = list(greedy_match(S, G1, G2))
    ground_truth = list(zip(alignment.iloc[:, 1], alignment.iloc[:, 0]))
    train = list(zip(seed_list1, seed_list2))
    ss = [str(x) for x in ss]
    ground_truth = [str(x) for x in ground_truth]
    train = [str(x) for x in train]
    ss = list(set(ss).difference(set(train)))
    ground_truth = list(set(ground_truth).difference(set(train)))

    return 100 * len(np.intersect1d(ss, ground_truth)) / len(ground_truth)


def one2one_accuracy(S, G1, G2, alignment):
    ss = list(greedy_match(S, G1, G2))
    ground_truth = list(zip(alignment.iloc[:, 1], alignment.iloc[:, 0]))
    ss = [str(x) for x in ss]
    ground_truth = [str(x) for x in ground_truth]
    return 100 * len(np.intersect1d(ss, ground_truth)) / len(ground_truth)


def topk_accuracy(S, G1, G2, alignment_dict_reversed, k):
    G2_nodes = list(G2.nodes())
    argsort = np.argsort(-S, axis=1)
    G1_dict = {}
    for key, value in enumerate(list(G1.nodes())):
        G1_dict[value] = key
    G2_dict = {}
    for key, value in enumerate(list(G2.nodes())):
        G2_dict[value] = key
    L = []
    for i in range(len(argsort)):
        index = alignment_dict_reversed.get(G2_nodes[i], None)
        if index == None:
            continue
        L.append(np.where(argsort[i, :] == G1_dict[index])[0][0] + 1)
    return np.sum(np.array(L) < k) / len(L) * 100


def topk_accuracy_supervised(S, G1, G2, alignment_dict_reversed, k, seed_list1, seed_list2):
    G2_nodes = list(set(list(alignment_dict_reversed.keys())) - set(seed_list2))
    G1_nodes = list(set(list(alignment_dict_reversed.values())) - set(seed_list1))
    S = pd.DataFrame(S, index=list(G2.nodes()), columns=list(G1.nodes()))
    S = np.array(S.loc[G2_nodes, G1_nodes])
    argsort = np.argsort(-S, axis=1)
    G1_dict = {}
    for key, value in enumerate(G1_nodes):
        G1_dict[value] = key
    G2_dict = {}
    for key, value in enumerate(G2_nodes):
        G2_dict[value] = key
    L = []
    for i in range(len(argsort)):
        L.append(np.where(argsort[i, :] == G1_dict[alignment_dict_reversed[G2_nodes[i]]])[0][0] + 1)
    return np.sum(np.array(L) < k) / len(L) * 100


def prior_alignment(W1, W2):
    H = np.zeros([len(W2), len(W1)])
    d1 = np.sum(W1, 0)
    d2 = np.sum(W2, 0)
    for i in range(len(W2)):
        H[i, :] = np.abs(d2[i] - d1) / np.max(np.concatenate([np.array([d2[i]] * len(d1)), d1]).reshape(2, -1), axis=0)
    H = 1 - H
    return H


def split_graph(edge_tuples, orig_num_cc, cut_size, G):
    g = G.copy()
    np.random.shuffle(edge_tuples)
    test_edges = []
    k = 0
    for edge in edge_tuples:
        print('\r{}/{}'.format(k, cut_size), end='')
        node1, node2 = edge[0], edge[1]

        g.remove_edge(node1, node2)
        if nx.number_connected_components(g) > orig_num_cc:
            g.add_edge(node1, node2)
            continue
        k += 1
        test_edges.append([node1, node2])
        if k > cut_size:
            break
    return g, test_edges


def cal_degree_dict(G_list, G, layer):
    """

    @param G_list:
    @param G:
    @param layer:
    @return: dictionary of dict[layer_number][node_name] = { set of degree of each neighbor in layer_number }
    """
    G_degree = G.degree()
    degree_dict = {}
    degree_dict[0] = {}
    for node in G_list:
        degree_dict[0][node] = {node}
    for i in range(1, layer + 1):
        degree_dict[i] = {}
        for node in G_list:
            neighbor_set = []
            for neighbor in degree_dict[i - 1][node]:
                neighbor_set += nx.neighbors(G, neighbor)
            neighbor_set = set(neighbor_set)
            for j in range(i - 1, -1, -1):
                neighbor_set -= degree_dict[j][node]
            degree_dict[i][node] = neighbor_set
    for i in range(layer + 1):
        for node in G_list:
            if len(degree_dict[i][node]) == 0:
                degree_dict[i][node] = [0]
            else:
                degree_dict[i][node] = node_to_degree(G_degree, degree_dict[i][node])
    return degree_dict


def seed_link(seed_list1, seed_list2, G1, G2, anchor):
    k = 0
    for i in range(len(seed_list1) - 1):
        for j in range(np.max([anchor + 1, i + 1]), len(seed_list1)):
            if G1.has_edge(seed_list1[i], seed_list1[j]) and not G2.has_edge(seed_list2[i], seed_list2[j]):
                G2.add_edges_from([[seed_list2[i], seed_list2[j]]])
                k += 1
            if not G1.has_edge(seed_list1[i], seed_list1[j]) and G2.has_edge(seed_list2[i], seed_list2[j]):
                G1.add_edges_from([[seed_list1[i], seed_list1[j]]])
                k += 1
    print('Add seed links : {}'.format(k), end='\t')
    return k


def node_to_degree(G_degree, SET):
    SET = list(SET)
    SET = sorted([G_degree[x] for x in SET])
    return SET


def caculate_jaccard_coefficient(G1, G2, seed_list1, seed_list2, index, columns, alignment_dict=None):
    mul = int(np.max([np.max(G1.nodes()), np.max(G2.nodes())]))
    seed1_dict = {}
    seed1_dict_reversed = {}
    seed2_dict = {}
    seed2_dict_reversed = {}
    for i in range(len(seed_list1)):
        seed1_dict[i + 2 * (mul + 1)] = seed_list1[i]
        seed1_dict_reversed[seed_list1[i]] = i + 2 * (mul + 1)
        seed2_dict[i + 2 * (mul + 1)] = seed_list2[i] + mul + 1
        seed2_dict_reversed[seed_list2[i] + mul + 1] = i + 2 * (mul + 1)
    G1_edges = pd.DataFrame(G1.edges())
    G1_edges.iloc[:, 0] = G1_edges.iloc[:, 0].apply(lambda x: to_seed(x, seed1_dict_reversed))
    G1_edges.iloc[:, 1] = G1_edges.iloc[:, 1].apply(lambda x: to_seed(x, seed1_dict_reversed))
    G1_edges.iloc[:, 0] = G1_edges.iloc[:, 0].apply(lambda x: to_seed(x, seed1_dict_reversed))
    G2_edges = pd.DataFrame(G2.edges())
    G2_edges += mul + 1
    G2_edges.iloc[:, 0] = G2_edges.iloc[:, 0].apply(lambda x: to_seed(x, seed2_dict_reversed))
    G2_edges.iloc[:, 1] = G2_edges.iloc[:, 1].apply(lambda x: to_seed(x, seed2_dict_reversed))
    adj = nx.Graph()
    adj.add_edges_from(np.array(G1_edges))
    adj.add_edges_from(np.array(G2_edges))
    jaccard_dict = {}
    for G1_node in index:
        for G2_node in columns:
            if (G1_node, G2_node) not in jaccard_dict.keys():
                jaccard_dict[G1_node, G2_node] = 0
            jaccard_dict[G1_node, G2_node] += calculate_adj(adj.neighbors(G1_node), adj.neighbors(G2_node + mul + 1))

    jaccard_dict = [[x[0][0], x[0][1], x[1]] for x in jaccard_dict.items()]
    adj_matrix = np.array(jaccard_dict)
    return adj_matrix


def flatten(input_list):
    output_list = []
    while True:
        if input_list == []:
            break
        for index, i in enumerate(input_list):

            if type(i) == list:
                input_list = i + input_list[index + 1:]
                break
            else:
                output_list.append(i)
                input_list.pop(index)
                break

    return output_list


def clip(x):
    if x <= 0:
        return 0
    else:
        return x


def calculate_adj(setA, setB):
    setA = set(setA)
    setB = set(setB)
    ep = 0.5
    inter = len(setA & setB) + ep
    union = len(setA | setB) + ep

    adj = inter / union
    return adj


def to_seed(x, dictionary):
    try:
        return dictionary[x]
    except:
        return x


def edge_sample(G):
    edges = list(G.edges())
    test_edges_false = []
    while len(test_edges_false) < G.number_of_edges():
        node1 = np.random.choice(G.nodes())
        node2 = np.random.choice(G.nodes())
        if node1 == node2:
            continue
        if G.has_edge(node1, node2):
            continue
        test_edges_false.append([min(node1, node2), max(node1, node2)])
    edges = edges + test_edges_false
    return edges


def edge_sample_evaluate(G, ratio):
    edges = list(G.edges())
    test_edges_false = []
    while len(test_edges_false) < G.number_of_edges():
        node1 = np.random.choice(G.nodes())
        node2 = np.random.choice(G.nodes())
        if node1 == node2:
            continue
        if G.has_edge(node1, node2):
            continue
        test_edges_false.append([min(node1, node2), max(node1, node2)])

    random.shuffle(edges)
    random.shuffle(test_edges_false)

    train_edges_true = edges[:int(len(edges) * ratio)]
    test_edges_true = edges[int(len(edges) * ratio):]
    train_edges_false = test_edges_false[:int(len(test_edges_false) * ratio)]
    test_edges_false = test_edges_false[int(len(test_edges_false) * ratio):]

    return train_edges_true, train_edges_false, test_edges_true, test_edges_false


def seed_link_lr(emb, G1, G2, seed_list1, seed_list2, mul, test_edges_final1, test_edges_final2, alignment_dict,
                 alignment_dict_reversed):
    train_edges_G1 = edge_sample(G1)
    embedding1 = [np.concatenate([emb[list(G1.nodes()).index(edge[0])], emb[list(G1.nodes()).index(edge[1])],
                                  emb[list(G1.nodes()).index(edge[0])] * emb[list(G1.nodes()).index(edge[1])]]) for edge
                  in train_edges_G1]
    label1 = [1] * G1.number_of_edges() + [0] * (len(train_edges_G1) - G1.number_of_edges())

    train_edges_G2 = edge_sample(G2)
    embedding2 = [np.concatenate(
        [emb[list(G2.nodes()).index(edge[0]) + len(G1.nodes())], emb[list(G2.nodes()).index(edge[1]) + len(G1.nodes())],
         emb[list(G2.nodes()).index(edge[0]) + len(G1.nodes())] * emb[
             list(G2.nodes()).index(edge[1]) + len(G1.nodes())]]) for edge in
                  train_edges_G2]
    label2 = [1] * G2.number_of_edges() + [0] * (len(train_edges_G2) - G2.number_of_edges())

    embedding = embedding1 + embedding2
    label = label1 + label2

    edge_classifier = LogisticRegression(solver='liblinear', random_state=0)
    edge_classifier.fit(np.array(embedding), label)

    test_edges1 = []
    test_edges2 = []
    for i in range(len(seed_list1)):
        for j in range(i + 1, len(seed_list1)):
            if not G1.has_edge(seed_list1[i], seed_list1[j]) and G2.has_edge(seed_list2[i], seed_list2[j]):
                test_edges1.append([min(seed_list1[i], seed_list1[j]), max(seed_list1[i], seed_list1[j])])
            if not G2.has_edge(seed_list2[i], seed_list2[j]) and G1.has_edge(seed_list1[i], seed_list1[j]):
                test_edges2.append([min(seed_list2[i], seed_list2[j]), max(seed_list2[i], seed_list2[j])])
    test_edges1, test_edges2 = np.array(test_edges1), np.array(test_edges2)
    embedding1 = [np.concatenate([emb[list(G1.nodes()).index(edge[0])], emb[list(G1.nodes()).index(edge[1])],
                                  emb[list(G1.nodes()).index(edge[0])] * emb[list(G1.nodes()).index(edge[1])]]) for edge
                  in test_edges1]
    embedding2 = [np.concatenate(
        [emb[list(G2.nodes()).index(edge[0]) + len(G1.nodes())], emb[list(G2.nodes()).index(edge[1]) + len(G1.nodes())],
         emb[list(G2.nodes()).index(edge[0]) + len(G1.nodes())] * emb[
             list(G2.nodes()).index(edge[1]) + len(G1.nodes())]]) for edge in
                  test_edges2]
    val_preds = []
    val_labels = []
    if len(embedding1) != 0:
        val_preds1 = edge_classifier.predict_proba(embedding1)[:, 1]
        pred1 = test_edges1[val_preds1 > 0.5]
        actual_list1 = list([[alignment_dict[edge[0]], alignment_dict[edge[1]]] for edge in test_edges1])
        actual_list1 = list(G2.has_edge(edge[0], edge[1]) for edge in actual_list1)
        val_labels = actual_list1
        val_preds = list(val_preds1)
    else:
        pred1 = []
    if len(embedding2) != 0:
        val_preds2 = edge_classifier.predict_proba(embedding2)[:, 1]
        val_preds += list(val_preds2)
        actual_list2 = list(
            [[alignment_dict_reversed[edge[0]], alignment_dict_reversed[edge[1]]] for edge in test_edges2])
        actual_list2 = list(G1.has_edge(edge[0], edge[1]) for edge in actual_list2)
        val_labels += actual_list2
        pred2 = test_edges2[val_preds2 > 0.5]
    else:
        pred2 = []

    if len(val_preds) != 0:
        if len(val_labels) != 0:
            try:
                roc_score = roc_auc_score(val_labels, val_preds)
                print('ROC SCORE : {}'.format(roc_score))
            except Exception as ee:
                print('ROC SCORE EXCEPTION')
                print(ee)
        else:
            print('Empty val_labels')
    else:
        print('Empty val_preds')

    '''
    pred1 = [(alignment_dict[edge[0]], alignment_dict[edge[1]]) for edge in pred1]
    pred2 = [(alignment_dict_reversed[edge[0]], alignment_dict_reversed[edge[1]]) for edge in pred2]
    if len(test_edges_final1) != 0:
        test_edges_final1 = [(min([x[0], x[1]]), max([x[0], x[1]])) for x in test_edges_final1]
        #test_edges_final1 = [str(x) for x in test_edges_final1]
        precision1 = 100 * len(np.intersect1d(test_edges_final1, pred2))/len(test_edges_final1)
    else:
        precision1 = 0
    if len(test_edges_final2) != 0:
        test_edges_final2 = [(min([x[0], x[1]]), max([x[0], x[1]])) for x in test_edges_final2]
        #test_edges_final2 = [str(x) for x in test_edges_final2]
        precision2 = 100 * len(np.intersect1d(test_edges_final2, pred1))/len(test_edges_final2)
    else:
        precision2 = 0
    pred1 = [(alignment_dict_reversed[edge[0]], alignment_dict_reversed[edge[1]]) for edge in pred1]
    pred2 = [(alignment_dict[edge[0]], alignment_dict[edge[1]]) for edge in pred2]
    '''
    return pred1, pred2


def evaluate_lp(G1, G2, emb, mul):
    G1_train_edges_true, G1_train_edges_false, G1_test_edges_true, G1_test_edges_false = edge_sample_evaluate(G1, 0.75)
    embedding1 = [np.concatenate([emb[list(G1.nodes()).index(edge[0])], emb[list(G1.nodes()).index(edge[1])],
                                  emb[list(G1.nodes()).index(edge[0])] * emb[list(G1.nodes()).index(edge[1])]]) for edge
                  in G1_train_edges_true]
    embedding1 += [np.concatenate([emb[list(G1.nodes()).index(edge[0])], emb[list(G1.nodes()).index(edge[1])],
                                   emb[list(G1.nodes()).index(edge[0])] * emb[list(G1.nodes()).index(edge[1])]]) for
                   edge in G1_train_edges_false]
    label1 = [1] * len(G1_train_edges_true) + [0] * len(G1_train_edges_false)

    G2_train_edges_true, G2_train_edges_false, G2_test_edges_true, G2_test_edges_false = edge_sample_evaluate(G2, 0.75)
    embedding2 = [np.concatenate(
        [emb[list(G2.nodes()).index(edge[0]) + len(G1.nodes())], emb[list(G2.nodes()).index(edge[1]) + len(G1.nodes())],
         emb[list(G2.nodes()).index(edge[0]) + len(G1.nodes())] * emb[
             list(G2.nodes()).index(edge[1]) + len(G1.nodes())]]) for edge in G2_train_edges_true]
    embedding2 += [np.concatenate(
        [emb[list(G2.nodes()).index(edge[0]) + len(G1.nodes())], emb[list(G2.nodes()).index(edge[1]) + len(G1.nodes())],
         emb[list(G2.nodes()).index(edge[0]) + len(G1.nodes())] * emb[
             list(G2.nodes()).index(edge[1]) + len(G1.nodes())]]) for edge in G2_train_edges_false]
    label2 = [1] * len(G2_train_edges_true) + [0] * len(G2_train_edges_false)

    embedding = embedding1 + embedding2
    label = label1 + label2

    edge_classifier = LogisticRegression(solver='liblinear', random_state=0)
    edge_classifier.fit(np.array(embedding), label)

    # testing
    embedding1 = [np.concatenate([emb[list(G1.nodes()).index(edge[0])], emb[list(G1.nodes()).index(edge[1])],
                                  emb[list(G1.nodes()).index(edge[0])] * emb[list(G1.nodes()).index(edge[1])]]) for edge
                  in G1_test_edges_true]
    embedding1 += [np.concatenate([emb[list(G1.nodes()).index(edge[0])], emb[list(G1.nodes()).index(edge[1])],
                                   emb[list(G1.nodes()).index(edge[0])] * emb[list(G1.nodes()).index(edge[1])]]) for
                   edge in G1_test_edges_false]
    label1 = [1] * len(G1_test_edges_true) + [0] * len(G1_test_edges_false)

    embedding2 = [np.concatenate(
        [emb[list(G2.nodes()).index(edge[0]) + len(G1.nodes())], emb[list(G2.nodes()).index(edge[1]) + len(G1.nodes())],
         emb[list(G2.nodes()).index(edge[0]) + len(G1.nodes())] * emb[
             list(G2.nodes()).index(edge[1]) + len(G1.nodes())]]) for edge in G2_test_edges_true]
    embedding2 += [np.concatenate(
        [emb[list(G2.nodes()).index(edge[0]) + len(G1.nodes())], emb[list(G2.nodes()).index(edge[1]) + len(G1.nodes())],
         emb[list(G2.nodes()).index(edge[0]) + len(G1.nodes())] * emb[
             list(G2.nodes()).index(edge[1]) + len(G1.nodes())]]) for edge in G2_test_edges_false]
    label2 = [1] * len(G2_test_edges_true) + [0] * len(G2_test_edges_false)

    embedding = embedding1 + embedding2
    label = label1 + label2

    val_preds = edge_classifier.predict_proba(embedding)[:, 1]
    auc_score = roc_auc_score(label, val_preds)
    print('LP AUC Score : {}'.format(auc_score), end='\t')

    for i in range(len(val_preds)):
        val_preds[i] = round(val_preds[i])
    precision = precision_score(label, val_preds)
    recall = recall_score(label, val_preds)
    print('LP PRECISION : {}'.format(precision), end='\t')
    print('LP RECALL : {}'.format(recall), end='\t')
