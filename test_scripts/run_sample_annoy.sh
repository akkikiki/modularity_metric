DIM=100
WORD_VEC=data/eng_jpn_sample_embedding.txt
k=3
TREE=450
echo "k=$k"
echo "tree=$TREE"
python3 src/modularity.py \
    --w2v $WORD_VEC \
    --tgt_lang jpn \
    --topk $k \
    --dim $DIM \
    --annoy
