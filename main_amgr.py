import os
from pprint import pprint
from tqdm import tqdm
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import utils
from model import resnet32
from config import get_arguments
import numpy as np
import math
import time
import torch.nn.functional as F
import json
from torchvision.utils import save_image


parser = get_arguments()
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.device = device
exp_loc, model_loc = utils.log_folders(args)
writer = SummaryWriter(log_dir=exp_loc)
score = {}


def main():
    """Main script"""

    assert not (args.logit_adj_post and args.logit_adj_train)
    # train_dataset, val_loader, num_train = utils.get_loaders(args)
    train_loader, val_loader, num_train= utils.get_loaders_v2(args)

    num_class = len(args.class_names)
    model = torch.nn.DataParallel(resnet32(num_classes=num_class))
    # model = resnet32(num_classes=num_class)

    model = model.to(device)
    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss(reduction='none').to(device)
    
    ####create z initialization#########
    z = np.zeros(num_train)

    gamma = args.gamma
 

    if args.logit_adj_post:
        if os.path.isfile(os.path.join(model_loc, "model.th")):
            print("=> loading pretrained model ")
            checkpoint = torch.load(os.path.join(model_loc, "model.th"))
            model.load_state_dict(checkpoint['state_dict'])
            for tro in args.tro_post_range:
                args.tro = tro
                args.logit_adjustments = utils.compute_adjustment(train_loader, tro, args)
                val_loss, val_acc = validate(val_loader, model, criterion)
                results = utils.class_accuracy(val_loader, model, args)
                results["OA"] = val_acc
                pprint(results)
                hyper_param = utils.log_hyperparameter(args, tro)
                writer.add_hparams(hparam_dict=hyper_param, metric_dict=results)
                writer.close()
        else:
            print("=> No pre trained model found")

        return

   

    args.logit_adjustments = utils.compute_adjustment(train_loader, args.tro_train, args)

    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=args.scheduler_steps)

    loop = tqdm(range(0, args.epochs), total=args.epochs, leave=False)
    val_loss, val_acc = 0, 0
    for epoch in loop:
         # train for one epoch
        # train_loss, train_acc = train(train_dataset, model, criterion, optimizer,num_train,gamma,z,epoch)
        train_loss, train_acc = train_v2(train_loader, model, criterion, optimizer, num_train, gamma, z, epoch,compute_loss)
        writer.add_scalar("train/acc", train_acc, epoch)
        writer.add_scalar("train/loss", train_loss, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        if (epoch % args.log_val) == 0 or (epoch == (args.epochs - 1)):
            val_loss, val_acc = validate(val_loader, model, criterion)
            writer.add_scalar("val/acc", val_acc, epoch)
            writer.add_scalar("val/loss", val_loss, epoch)

        loop.set_description(f"Epoch [{epoch}/{args.epochs}")
        loop.set_postfix(train_loss=f"{train_loss:.2f}", val_loss=f"{val_loss:.2f}",
                         train_acc=f"{train_acc:.2f}",
                         val_acc=f"{val_acc:.2f}")

        class_cnt = [0]*10
        records = []
        for _, (inputs, target,idx) in enumerate(train_loader):
             
            for i, index in enumerate(idx): 
                index = int(index)
                record = [index, score[index].cpu(), int(target[i])]
                records.append(record)
                
                # class_name = int(target[i])
                # class_cnt[class_name] += 1
                # image_name = args.save_dir+'/label_'+ str(class_name) + '_' + str(class_cnt[class_name]) +'_'+str(score[item]) + '.png'
                # save_image(item, image_name)
        records = np.array(records)
        with open('score.npy', 'wb') as f:
            np.save(f, records)
        print('finish saving')

    file_name = 'model.th'
    mdel_data = {"state_dict": model.state_dict()}
    torch.save(mdel_data, os.path.join(model_loc, file_name))

    results = utils.class_accuracy(val_loader, model, args)
    results["OA"] = val_acc
    hyper_param = utils.log_hyperparameter(args, args.tro_train)
    pprint(results)
    writer.add_hparams(hparam_dict=hyper_param, metric_dict=results)
    writer.close()


     
   


def compute_grad(sample, target, criterion, model):
    # start_time = time.time()
    prediction = model(sample)
    loss = criterion(prediction, target)

    for i in range(sample.shape[0]):
        data,label = sample[i],target[i]
        grad = torch.autograd.grad(loss[i],  list(model.parameters())[-1],retain_graph=True )

        grad = grad[0].flatten().unsqueeze(0)
        if i == 0:
            grads = grad
        else:
            grads = torch.cat([grads,grad],dim=0)

    # print("---compute_grad runtime is %s seconds ---" % (time.time() - start_time))
 
    return grads

#chatgpt version solution
# def compute_per_sample_gradients(model, x, target,criterion):
#     # Ensure model is in training mode
#     model.train()

#     # Register hook on the last_layer of the model
#     gradients = []
#     def hook_function(module, grad_input, grad_output):
#         gradients.append(grad_input[0])
#     hook = model.linear.register_backward_hook(hook_function)

#     # Forward pass
#     output = model(x)

#     # Compute the loss
#     loss = criterion(output, target)

#     # Backward pass
#     model.zero_grad()
#     for single_loss in loss:
#         single_loss.backward(retain_graph=True)

#     # Remove the hook
#     hook.remove()

#     # Now, gradients holds the per-sample gradients of the weights in the last_layer
#     return gradients

# my solution
def compute_per_sample_gradients(model, x, target,criterion):

    with torch.no_grad():  
        features = model(x,layer = 1)

    for i, f in enumerate(features): 
  
        loss = criterion(model.module.linear(f), target[i])
        

        loss = loss.mean()
        grad = torch.autograd.grad(loss,  list(model.parameters())[-1],retain_graph=True )

        grad = grad[0].flatten().unsqueeze(0)
        if i == 0:
            grads = grad
        else:
            grads = torch.cat([grads,grad],dim=0)
 

    # Now, gradients holds the per-sample gradients of the weights in the last_layer
    return grads


def compute_loss(params,  buffers, sample, target,model,criterion):
    batch = sample.unsqueeze(0)
    targets = target.unsqueeze(0)


 
    predictions = functional_call(model, (params, buffers), (batch,))
    loss = criterion(predictions, targets)
    return loss.mean()





def q(model,criterion,grad_i,x_j,y_j,gamma):
    # start_time = time.time()

    cos = torch.nn.CosineSimilarity(dim=0)
   
    # grad_i = compute_grad(x_i, y_i,criterion, model)
     

    grad_j = compute_grad(x_j, y_j, criterion,model) 
 
    arr = np.arange(len(grad_i)) 
 
    np.random.shuffle(arr)
    corr = 0
    # for i in range(int(len(arr)*0.01)):
    #     corr = corr + cos( grad_i[arr[i]].flatten(), grad_j[arr[i]].flatten() )
    with torch.no_grad():

        corr = cos( grad_i[-1].flatten(), grad_j[-1].flatten() )
    # print("---q runtime is %s seconds ---" % (time.time() - start_time))

    return max( corr-gamma ,0 )

def embedding_corr(model, output_i, x_j,gamma):
    cos = torch.nn.CosineSimilarity(dim=0)
    x_j = x_j.unsqueeze(0)  # prepend batch dimension for processing
    output = model(x_j)
    output_j = output[-1]
    corr = cos(output_i.flatten(), output_j.flatten())
    return max( corr-gamma ,0 ) 



def weighted_criterion(outputs,labels,criterion,weight):
    # start_time = time.time()

    weighted_loss = torch.tensor(0)
    weighted_loss.to(device)
    for i in range(len(outputs)):
        weighted_loss = weighted_loss + weight[i]*criterion(outputs[i],labels[i])
 
    # print("---weighted_criterion runtime is %s seconds ---" % (time.time() - start_time))

    return weighted_loss 


def train_v2(train_loader, model, criterion, optimizer, num_train, gamma, z, epoch,compute_loss):
    """ Run one train epoch """

    losses = utils.AverageMeter()
    accuracies = utils.AverageMeter()

    model.train()
    
 
    for _, (inputs, target,idx) in enumerate(train_loader):
        target = target.to(device)
        input_var = inputs.to(device)
        target_var = target
  
                
        output = model(input_var)

        if args.logit_adj_train:
            output = output + args.logit_adjustments
        weighted_loss = 0
        if args.amgr:
            grads = compute_per_sample_gradients(model, input_var, target_var,criterion)
            if args.norm:
                grads = F.normalize(grads,p=2.0) 

            grads_t = torch.transpose(grads, 0, 1)
            if args.temp_decay:
                temp = args.temp*(epoch/100+1)
            else:
                temp = args.temp
            gram = torch.matmul(grads,grads_t) 
            gram = gram - args.off_diag*torch.eye(gram.size(0)).to(device)
            gram = F.relu(torch.sub(gram,gamma))
            weights = torch.sum(gram, 1)
            for i, index in enumerate(idx):
                index = int(index)
                if index in score:
                    weights[i] = weights[i]+score[index]
                    score[index]= weights[i]
                else:
                    score[index] = weights[i]

            weights = weights/(temp*(epoch+1))
            weights = F.softmax(-weights)
            weights = weights.detach()
            
            if args.measure == 1:
                features = model(x,layer = 1)
                features = F.normalize(features,p=2.0)
                features_t = torch.transpose(features, 0, 1)
                ft_gram = torch.matmul(features,features_t)
                ft_gram = F.relu(torch.sub(ft_gram,gamma))
                ft_weights = torch.sum(ft_gram, 1)
                ft_weights = ft_weights/temp
                ft_weights = F.softmax(-ft_weights)
                ft_weights = ft_weights.detach()
                alpha = epoch/1241
                weights = (1-alpha)*weights + alpha*ft_weights
                weights.detach()
                if args.attn:
                    weighted_loss = torch.matmul(F.softmax(-gram, dim = 1), loss ).mean()
                else:
                    weighted_loss = torch.inner(loss,weights)
       
        acc = utils.accuracy(output.data, target)

        loss = criterion(output, target_var)
        
        

        loss=loss.mean()
        loss_r = 0
        for parameter in model.parameters():
            loss_r += torch.sum(parameter ** 2)
        loss = loss + args.weight_decay * loss_r
        if args.amgr:
            weighted_loss = weighted_loss + args.weight_decay * loss_r

        optimizer.zero_grad()
        if args.amgr:
            weighted_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

    return losses.avg, accuracies.avg



 


def validate(val_loader, model, criterion):
    """ Run evaluation """

    losses = utils.AverageMeter()
    accuracies = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for _, (inputs, target,idx) in enumerate(val_loader):
           
            target = target.to(device)
            input_var = inputs.to(device)
            target_var = target.to(device)

            output = model(input_var)
            loss = criterion(output, target_var)
            loss = loss.mean()
            if args.logit_adj_post:
                output = output - args.logit_adjustments

            elif args.logit_adj_train:
                loss = criterion(output + args.logit_adjustments, target_var)
            loss = loss.mean()

            acc = utils.accuracy(output.data, target)
            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))

    return losses.avg, accuracies.avg


if __name__ == '__main__':
    main()
