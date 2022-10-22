# GNN Jet Autoencoder
A GNN autoencoder for jets in particle physics implemented in PyTorch.

## Data
To download data:
1. Install `JetNet`:
    ```
    pip3 install jetnet; 
    ```
2. Run `preprocess.py`
    ```
    python utils/data/preprocess.py \
    --jet-types g q t w z \
    --save-dir "./data"
    ```

## Training
To train the model, run `train.py`. An example is provided in `examples/main.sh`.