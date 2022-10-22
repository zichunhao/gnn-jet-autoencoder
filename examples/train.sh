#!/bin/bash

mkdir -p "dev";
python train.py \
--batch-size 16 \
--jet-type qcd \
--data-paths "data/g_jets_30p_small.pt" "data/q_jets_30p_small.pt" \
--encoder-edge-sizes '16,16,8,8;' \
--decoder-edge-sizes '16,16,8,8;' \
--encoder-node-sizes '3;3;3;3;' \
--decoder-node-sizes '3;3;3;3;' \
--save-dir dev \
--num-epochs 10 \
--latent-map "local mix" \
--latent-node-size 2
--train-fraction 0.65 \
--save-dir dev \
| tee -a dev/autoencoder-g-s1-v1.txt
