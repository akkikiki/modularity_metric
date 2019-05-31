import argparse
import numpy as np
import pickle
from utils import load_sparse_csr
from utils_lex_graph import build_graph

def calc_weighted_modularity_langs(A_filename, indice2word_filename, langs):
    """

    :param A_filename:
    :param indice2word_filename:
    :param langs: prefix tags for each word in the .txt embedding file
    :return: normalized modularity
    """
    A = load_sparse_csr(A_filename).todok()
    indice2word = pickle.load(open(indice2word_filename, "rb"))
    num_community = len(langs)
    a_l = [0] * num_community
    e_ll = [0] * num_community
    e_ll_Q_max = [0] * num_community
    lang_to_indice = {lang: i for i, lang in enumerate(langs)}  # assigns a community ID for each language

    degree_list, two_m = get_degrees(A)

    """
    a_l computation
    """
    for node_id in range(A.shape[0]):
        node_word = indice2word[node_id]
        community = lang_to_indice[node_word[:3]]
        k_i = degree_list[node_id]
        a_l[community] += k_i

    """
    e_ll computation
    """
    for processed, e in enumerate(A.items()):
        node1, node2 = e[0]
        node1_word = indice2word[node1]
        node2_word = indice2word[node2]
        community_id = lang_to_indice[node1_word[:3]]

        #e_ll_Q_max[community_id] += A[node1, node2]
        e_ll_Q_max[community_id] += 1 # Since 1 is maximum possible weight
        if node1_word[:3] == node2_word[:3]:
            e_ll[community_id] += A[node1, node2] # counting both directions
        
    check_e_ii_s_Q_max = 0
    for i in range(len(a_l)):
        check_e_ii_s_Q_max += e_ll_Q_max[i]
    assert(check_e_ii_s_Q_max == two_m)
 
        
    e_ll = list(map(lambda x:x/two_m, e_ll))
    e_ll_Q_max = list(map(lambda x:x/two_m, e_ll_Q_max))
    a_l = list(map(lambda x:x/two_m, a_l))
    print("mean degree=%.4f"% (np.mean(degree_list)))

    Q = 0
    Q_max = 0
    for i in range(len(a_l)):
        Q += e_ll[i] - (a_l[i] ** 2)
        Q_max += e_ll_Q_max[i] - (a_l[i] ** 2)
    print("Weighted Q=%.4f" % Q)
    print("Weighted Q_max=%.4f" % Q_max)
    print("Normalized Q=%.4f" % (Q/Q_max))
    return Q/Q_max


def get_degrees(A):
    """
    Pre-computation for degree of each node in the adjacency matrix
    :param A: adjacency matrix
    :return: degrees for each node, 2 * number of edges
    """
    two_m = 0
    degree_list = [0] * A.shape[0]
    for processed, e in enumerate(A.items()):
        node1, node2 = e[0]
        degree_list[node1] += A[node1, node2]
        two_m += 1  # total num. of edges * 2
    return degree_list, two_m


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--w2v", help="Word embedding file")
    parser.add_argument("--src_lang", default="eng", help="language tag for the source language. e.g., 'jpn' ")
    parser.add_argument("--tgt_lang", required=True, help="language tag for the language other than English. e.g., 'jpn' ")
    parser.add_argument("--topk", type=int, help="k for k-NN graph", default=100)
    parser.add_argument("--annoy", help="Construct an approximate nearest neighbor graph using ANNOY",
                   default=False, action="store_true")
    parser.add_argument("--subgraph", help="Compute modularity on a nearest neighbor graph built from a subset of vocabularies",
                       default=False, action="store_true")
    parser.add_argument("--tree_num", help="Num. of trees used to get approximated nearest neighbors",
                       type=int, default=20)
    parser.add_argument("--save_dir", help="Temporary output dir to save the pickled graph", type=str,
                        default="pickles/")
    parser.add_argument("--dim", help="word embedding dimension", type=int, default=100)
    args = parser.parse_args()

    A_name = args.save_dir + "A.npy.npz"
    indice2word_name = args.save_dir + "indice2word.pickle"
    build_graph(args.w2v, args.topk, A_name, indice2word_name, annoy=args.annoy, dim=args.dim, tree_num=args.tree_num)

    langs_prefix= [args.src_lang, args.tgt_lang]

    Q_norm = calc_weighted_modularity_langs(A_name, indice2word_name, langs_prefix)
