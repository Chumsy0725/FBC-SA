# Towards Generalizing to Unseen Domains with Few Labels

This code is the official implementation of the following paper: [Towards Generalizing to Unseen Domains with Few Labels]().


## How to setup the environment

This code is built on top of [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch). Please follow the instructions provided in https://github.com/KaiyangZhou/Dassl.pytorch to install the `dassl` environment, as well as to prepare the datasets. 
The style augmentation is based on [AdaIN](https://arxiv.org/abs/1703.06868) and the implementation is based on this code https://github.com/naoto0804/pytorch-AdaIN. Please download the weights of the decoder and the VGG from https://github.com/naoto0804/pytorch-AdaIN and put them under a new folder `ssdg-benchmark/weights`.

## How to run

The script is provided in `ssdg-benchmark/scripts/FBASA/run_ssdg.sh`. You need to update the `DATA` variable that points to the directory where you put the datasets. There are three input arguments: `DATASET` and `NLAB` (total number of labels).


Here we give an example. Say you want to run StyleMatch on PACS under the 10-labels-per-class setting (i.e. 210 labels in total), simply run the following commands in your terminal,
```bash
conda activate dassl
cd ssdg-benchmark/scripts/FBCSA
bash run_ssdg.sh ssdg_pacs 210 
```

In this case, the code will run StyleMatch in four different setups (four target domains), each for five times (five random seeds). You can modify the code to run a single experiment instead of all at once if you have multiple GPUs.


To show the results, simply do
```bash
python parse_test_res.py output/ssdg_pacs/nlab_210/StyleMatch/resnet18/v1 --multi-exp
```


