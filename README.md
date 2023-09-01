# batch-reweighting-cifar
This is the code for batch reweighting training on CIFAR10/100 (LT).

## General usage of batch reweighting
Batch rewighting can be treated as a building block and incorporated into any training task. What we need to do is 

1) Write a per_sample_grad(model, inputs, targets, criterion) function, which takes model and mini-batch data as well as loss function as input argument, and outputs per sample gradients, a B by d tensor, where B is mini-batch size and d is dimension of gradient for each sample.

2) Adding following lines of code inside your train() function's loop. For example:
  
```python
  criterion = nn.CrossEntropyLoss(reduction='none').to(device) # here we want cross entropy loss to return per sample losses given a mini-batch, so we set reduction = 'none'
  for  _, (inputs, target) in enumerate(train_loader):
      #per sample gradient computation
      grads = compute_per_sample_gradients(model, inputs, target,criterion)

      # normalize and transpose
      grads = F.normalize(grads,p=2.0)  
      grads_t = torch.transpose(grads, 0, 1)
      gram = torch.matmul(grads,grads_t)

      # compute p score
      p = [(row>=gamma).sum() for row in gram]

      # compute weights
      weights = torch.tensor(weights).to(device)
      weights = weights/temp
      weights = weights.detach()

      # compute weighted loss and backpropagate weighted loss
      weighted_loss = torch.inner(loss,weights)   
      optimizer.zero_grad()
      weighted_loss.backward()
      optimizer.step()

```


## Running the code 

In this repo we already integrate batch reweighting for standard CIFAR10/100-LT training. Here are usage:

1) First, you need to put cifar10/100-lt dataset (npz format) into ``data`` folder
2) Then run one of the following command

```python
# To produce baseline (ERM) results on cifar10-lt:
python main.py --dataset cifar10-lt

# To produce baseline (ERM) results on cifar10:
python main.py --dataset cifar10

# To produce batch reweighting results:
python main.py --dataset cifar10-lt  --br 1


# To monitor the training progress using Tensorboard:
tensorboard --logdir logs


```

## Saved score for each image

The score averaged throughout all epoches for each image is stored at ``` /batch-reweighting-cifar/scores/[Your running configuration]/score.npy ```.
It is stored into a numpy array, whose dimension is [# of data, 3], where first column indicates the file indices, second is its p score and third is its class.


## Important parameters

```bash
usage: main.py [-h] [-dataset Dataset] [--br BR] [--gamma Gamma] [--temp Temp] [--num_workers N] [--batch-size B] [--lr LR] [--momentum M] [--weight-decay W]

PyTorch Cifar Training

 arguments:
  -h, --help            show this help message and exit
  --dataset             dataset to train (default: cifar10-lt)
  --br                  whether to enable batch reweighting. 1 for enabling and 0 for normal training (default 0)
  --gamma               value of gamma. (default: 0.7)
  --temp                tempertuare parameter inside softmax. (default: 1)
  --num_workers N       number of data loading workers (default: 4)
  --batch-size B        mini-batch size (default: 128), this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel
  --lr LR               initial learning rate (default 0.1)
  --momentum M          momentum (default 0.9)
  --weight-decay W      weight decay (default: 1e-4)
 

```
