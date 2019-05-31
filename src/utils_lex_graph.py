from scipy.sparse import dok_matrix
import pickle
import numpy as np
from utils import read_w2v, save_sparse_csr

def build_graph(filename, TOPN, A_name, indice2word_name, annoy=False, dim=100, tree_num=20):
    """
    """
    model = read_w2v(filename, dim)
    V = len(model.wv.vocab)
    print("Num. vocab = %i" % V)
    word_indice_dic = {word: i for i, word in enumerate(model.wv.vocab)}
    indice2word = {i: word for word, i in word_indice_dic.items()}
    A = dok_matrix((V, V), dtype=np.float32)
    if annoy:
        print("Using ANNOY...")
        from gensim.similarities.index import AnnoyIndexer
        annoy_index = AnnoyIndexer(model, tree_num)
        add_neighbors(A, TOPN, model, word_indice_dic, annoy_index=annoy_index)
    else:
        add_neighbors(A, TOPN, model, word_indice_dic)

    save_sparse_csr(A_name, A.tocsr())
    pickle.dump(indice2word, open(indice2word_name , "wb"))


def add_neighbors(A, TOPN, model, word_indice_dic, annoy_index=None):
    for word, indice in word_indice_dic.items():
        finished = 0
        if annoy_index:
            word_sim_list = model.most_similar(positive=[word], topn=TOPN + 1, indexer=annoy_index)
        else:
            word_sim_list = model.most_similar(positive=[word], topn=TOPN)

        for sim_word, cos_sim in word_sim_list:
            target_indice = word_indice_dic[sim_word]
            if indice == target_indice:
                continue  # avoid adding self-loops
            A[indice, target_indice] = max(cos_sim, 0.0)
            A[target_indice, indice] = max(cos_sim, 0.0)
            finished += 1
