# Towards Generalizing to Unseen Domains with Few Labels

This code is the official implementation of the following paper: [Towards Generalizing to Unseen Domains with Few Labels]().


## How to setup the environment

This code is built on top of [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) and [ssdg-benchmark](https://github.com/KaiyangZhou/ssdg-benchmark). Please follow the instructions provided in https://github.com/KaiyangZhou/Dassl.pytorch and https://github.com/KaiyangZhou/ssdg-benchmark to install the `dassl` environment, as well as to prepare the datasets. 

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
python parse_test_res.py output/ssdg_pacs/nlab_210/FBCSA/resnet18 --multi-exp
```


