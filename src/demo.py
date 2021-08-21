# -*- coding: utf-8 -*-
import time

from CENALP import CENALP
import pandas as pd
from utils import *
import warnings
import argparse

warnings.filterwarnings('ignore')


def read_alignment(alignment_folder, filename):
    alignment = pd.read_csv(alignment_folder + filename + '.csv', header=None)
    alignment_dict = {}
    alignment_dict_reversed = {}
    for i in range(len(alignment)):
        alignment_dict[alignment.iloc[i, 0]] = alignment.iloc[i, 1]
        alignment_dict_reversed[alignment.iloc[i, 1]] = alignment.iloc[i, 0]
    return alignment_dict, alignment_dict_reversed


def read_attribute(attribute_folder, filename, G1, G2):
    try:
        attribute, attr1, attr2 = load_attribute(attribute_folder, filename, G1, G2)
        attribute = attribute.transpose()
    except:
        attr1 = []
        attr2 = []
        attribute = []
        print('Attribute files not found.')
    return attribute, attr1, attr2


def parse_args():
    '''
    Parses the CLF arguments.
    '''
    parser = argparse.ArgumentParser(description="Run CENALP.")
    parser.add_argument('--attribute_folder', nargs='?', default='../attribute/',
                        help='Input attribute path')

    parser.add_argument('--data_folder', nargs='?', default='../graph/',
                        help='Input graph data path')
    parser.add_argument('--alignment_folder', nargs='?', default='../alignment/',
                        help='Input ground truth alignment path')
    parser.add_argument('--filename', nargs='?', default='bigtoy',
                        help='Name of file')
    parser.add_argument('--alpha', type=int, default=5,
                        help="Hyperparameter controlling the distribution of vertices' similairties")
    parser.add_argument('--layer', type=int, default=3,
                        help="Depth of neighbors")
    parser.add_argument('--align_train_prop', type=float, default=0.0,
                        help="Proportion of training set. 0 represents unsupervised learning.")
    parser.add_argument('--q', type=float, default=0.5,
                        help="Probability of walking to the separate network during random walk")
    parser.add_argument('--c', type=float, default=0.5,
                        help="Weight between sub-graph similarity and attribute similarity")
    parser.add_argument('--multi_walk', type=bool, default=False,
                        help="Whether to use multi-processing")
    parser.add_argument('--neg_sampling', type=bool, default=False,
                        help="Use Negative Sampling")
    return parser.parse_args()

def generateXnodesFromGraph(G1,G2,alignment_dict,size):
    startingNodeIndex = random.randint(0, len(G1.nodes()) - 1)
    nodes = set()
    currentNode = list(G1.nodes())[startingNodeIndex]
    while len(nodes) < size:
        nodes.add(currentNode)
        toIndex = random.randint(0, len(list(G1.neighbors(currentNode))) - 1)
        currentNode = list(G1.neighbors(currentNode))[toIndex]

    with open('DBLP' + str(size) + '_1.edges', 'w') as file:
        for edge in G1.edges():
            if edge[0] in nodes and edge[1] in nodes:
                line = str(edge[0]) + ', ' + str(edge[1])
                line += '\n'
                file.write(line)

    with open('DBLP' + str(size) + '_.csv', 'w') as file:
        for node in nodes:
            line = str(node) + ', ' + str(alignment_dict[node])
            line += '\n'
            file.write(line)

    G2_nodes = []
    for node in nodes:
        G2_nodes.append(alignment_dict[node])

    with open('DBLP' + str(size) + '_2.edges', 'w') as file:
        for edge in G2.edges():
            if edge[0] in G2_nodes and edge[1] in G2_nodes:
                line = str(edge[0]) + ', ' + str(edge[1])
                line += '\n'
                file.write(line)
    print('ssssss')

def main(args):
    print(args)
    alignment_dict, alignment_dict_reversed = read_alignment(args.alignment_folder, args.filename)
    G1, G2 = loadG(args.data_folder, args.filename)

    # counter = 0
    # for edge in G1.edges():
    #     if G2.has_edge(alignment_dict.get(edge[0],-1),alignment_dict.get(edge[1],-1)):
    #         counter+=1
    # print('ddddddddddddd')
    # print((2*counter)/(len(G1.edges()) + len(G2.edges())))


    # generateXnodesFromGraph(G1, G2, alignment_dict, 0*400 + 500)
    # generateXnodesFromGraph(G1, G2, alignment_dict, 1*400 + 500)
    # generateXnodesFromGraph(G1, G2, alignment_dict, 2*400 + 500)
    # generateXnodesFromGraph(G1, G2, alignment_dict, 3*400 + 500)
    # generateXnodesFromGraph(G1, G2, alignment_dict, 4*400 + 500)
    # G1_nodes_list = list(G1.nodes())
    # G2_nodes_list = list(G2.nodes())
    # for node in G1_nodes_list:
    #     if not node in alignment_dict:
    #         G1.remove_node(node)
    #
    # for node in G2_nodes_list:
    #     if not node in alignment_dict_reversed:
    #         G2.remove_node(node)
    #
    # with open('111.txt', 'w') as file:
    #     for edge in G1.edges():
    #         line = str(edge[0]) + ', ' + str(edge[1])
    #         line += '\n'
    #         file.write(line)
    # with open('222.txt', 'w') as file:
    #     for edge in G2.edges():
    #         line = str(edge[0]) + ', ' + str(edge[1])
    #         line += '\n'
    #         file.write(line)
    # print(len(G1.nodes()))
    # print(len(G2.nodes()))
    attribute, attr1, attr2 = read_attribute(args.attribute_folder, args.filename, G1, G2)
    start_time = time.time()
    S, precision, seed_l1, seed_l2 = CENALP(G1, G2, args.q, attr1, attr2, attribute, alignment_dict, alignment_dict_reversed,
       args.layer, args.align_train_prop, args.alpha, args.c, args.multi_walk,args.neg_sampling)
    print("--- %s seconds ---" % (time.time() - start_time))
if __name__ == '__main__':
    args = parse_args()
    main(args)
