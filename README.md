# modularity_metric
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

## Example Usage
```
    sh run_sample.sh
```

## References
Yoshinari Fujinuma, Jordan Body-Graber, and Michael J. Paul, A Resource-Free Evaluation Metric for Cross-Lingual Word Embeddings based on Graph Modularity, ACL 2019
```
@inproceedings{clwe_modularity,
   title = "A Resource-Free Evaluation Metric for Cross-Lingual Word Embeddings based on Graph Modularity",
   author = "Fujinuma, Yoshinari AND Boyd-Graber, Jordan AND Paul, Michael J.",
   booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
   year = "2019",
}
```
