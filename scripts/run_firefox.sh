DIM=100
k=6
WORD_VEC=data/eng_jpn_mse_orth.txt
echo "k=$k"
python3 src/modularity.py \
    --w2v $WORD_VEC \
    --tgt_lang jpn \
    --topk $k \
    --dim $DIM \
    --subgraph \
    --firefox