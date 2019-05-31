import numpy as np
from scipy import sparse
import gensim
from gensim.models import word2vec

def read_w2v(target_w2v_file, dim):
    if target_w2v_file.endswith(".txt"):
        target_w2v_model_kv = gensim.models.KeyedVectors.load_word2vec_format(target_w2v_file, binary=False)
        target_w2v_model = word2vec.Word2Vec(size=dim)
        target_w2v_model.wv = target_w2v_model_kv
    else:
        en_model_kv = gensim.models.KeyedVectors.load_word2vec_format(target_w2v_file, binary=True)
        target_w2v_model = word2vec.Word2Vec(size=dim)
        target_w2v_model.wv = en_model_kv
    return target_w2v_model

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return sparse.csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])
