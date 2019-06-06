# modularity_metric
An adhoc tool/metric to diagnose whether the resulting cross-lingual word embedding is "mixed" w.r.t to its language. 

## Requirements
* gensim
* (optional) [annoy](https://github.com/spotify/annoy)
```
    pip3 install -r requirements.txt
```

Confirmed that it runs on 
* Python 3.6.5.
* gensim 3.4.0
* annoy 1.8.3

## Usage
```
python3 src/modularity.py --w2v YOUR_VECTOR --src_lang SRC_LANG --tgt_lang TGT_LANG
```
Currently, the input vector is assumed to be a concatenated cross-lingual embedding where each word has a prefix tag of three characters (i.e., ISO 639-2 Code), e.g., 
```
python3 src/modularity.py --w2v $WORD_VEC --src_lang eng --tgt_lang jpn
```
and an example of a word vector is `eng:the 0.123988 -0.0562252...`. 

### Run tests
```
sh scripts/run_test.sh
```

### Example usage
```
sh scripts/run_sample.sh
```

### Example usage with annoy (approximate nearest neighbors)
```
sh scripts/run_sample_annoy.sh
```

### Reproduce Figure 1 in the paper
```
sh scripts/get_sample_embedding.sh
sh scripts/run_eat.sh
sh scripts/run_firefox.sh
```

## References
If you use this code, please cite our paper.

Yoshinari Fujinuma, Jordan Body-Graber, and Michael J. Paul, [A Resource-Free Evaluation Metric for Cross-Lingual Word Embeddings based on Graph Modularity](https://arxiv.org/abs/1906.01926), ACL 2019
```
@inproceedings{clwe_modularity,
   title = "A Resource-Free Evaluation Metric for Cross-Lingual Word Embeddings based on Graph Modularity",
   author = "Fujinuma, Yoshinari and Boyd-Graber, Jordan and Paul, Michael J.",
   booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
   year = "2019",
}
```
