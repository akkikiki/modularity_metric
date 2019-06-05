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
Yoshinari Fujinuma, Jordan Body-Graber, and Michael J. Paul, A Resource-Free Evaluation Metric for Cross-Lingual Word Embeddings based on Graph Modularity, ACL 2019
```
@inproceedings{clwe_modularity,
   title = "A Resource-Free Evaluation Metric for Cross-Lingual Word Embeddings based on Graph Modularity",
   author = "Fujinuma, Yoshinari and Boyd-Graber, Jordan and Paul, Michael J.",
   booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
   year = "2019",
}
```
