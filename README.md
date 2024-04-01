# SpaceFormer
SpaceFormer is a Transformer based computational tool for spatial omics analysis. Given a set of adata files, this document demonstrates how one can train a SpaceFormer in a self-supervised manner.

* environment

A conda virtual environment is highly recommended when training SpaceFormer. After you are at the root path of this directory, use the following commands to set up a proper virtual environment:
```bash
conda create -n SpaceFormer python=3.8
conda activate SpaceFormer
pip install -r requirements.txt
```

* data

Create a new folder named 'data' at the root directory, and then put all h5ad files under it. The structure of the folder should be like:
```
--utils
--scripts
--data
    -adata1.h5ad
    -adata2.h5ad
    ...
```
After the h5ad files are rightly placed, change the corresponding strings to your own setting in the `utils/const.py` file.

*  scripts

In order to run the SpaceFormer data, you should write the proper bash scripts. Two examples, which are the cross validation version of two different types of models, are given in the `scripts` folder. Here are some key points:
* Change the `$input` variable into the number of genes of your own data.
* `$device` denotes the gpu id of your experiment.
* Tune the other hyperparameters. Their settings are the optimized ones for the data used in the paper, but they may not be the proper ones for your own data.

After writing the bash scripts, use the following command to train your SpaceFormer model:
```bash
cd scripts
bash SpaceFormer_pretrain.sh
```
Good luck!
