# batch-reweighting-imagenet
This is the code for batch reweighting training on CIFAR10/100 (LT).

## Running the code 

```python
# To produce baseline (ERM) results:
python main.py --dataset cifar10-lt

# To produce batch reweighting results:
python main.py --dataset cifar10-lt  --br 1


# To monitor the training progress using Tensorboard:
tensorboard --logdir logs


```

## Saved score for each image

The score averaged throughout all epoches for each image is stored at ``` /batch-reweighting-cifar/scores/[Your running configuration]/score.npy ```.
It is stored into a numpy array, whose dimension is [# of data, 3], where first column indicates the filename, second is its p score and third is its class.


## Important parameters

```bash
usage: main.py [-h] [-a ARCH] [-j N] [--epochs N] [--start-epoch N] [-b N] [--lr LR] [--momentum M] [--wd W] [-p N] [--resume PATH] [-e] [--pretrained] [--world-size WORLD_SIZE] [--rank RANK]
               [--dist-url DIST_URL] [--dist-backend DIST_BACKEND] [--seed SEED] [--gpu GPU] [--multiprocessing-distributed] [--dummy]
               [DIR]

PyTorch Cifar Training

 arguments:
  -h, --help            show this help message and exit
  --dataset             dataset to train (default: cifar10-lt)
  --br                  whether to enable batch reweighting. 1 for enabling and 0 for normal training (default 0)
  --gamma               value of gamma. (default: 0.7)
  --temp                tempertuare parameter inside softmax. (default: 1)
  --num_workers N       number of data loading workers (default: 4)
  --batch-size N        mini-batch size (default: 128), this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel
  --lr LR               initial learning rate (default 0.1)
  --momentum M          momentum (default 0.9)
  --weight-decay W      weight decay (default: 1e-4)
 

```
