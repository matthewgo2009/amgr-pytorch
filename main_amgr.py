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

parser = get_arguments()
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.device = device
exp_loc, model_loc = utils.log_folders(args)
writer = SummaryWriter(log_dir=exp_loc)


def main():
    """Main script"""

    assert not (args.logit_adj_post and args.logit_adj_train)
    train_dataset, val_loader, num_train = utils.get_loaders(args)
    num_class = len(args.class_names)
    model = torch.nn.DataParallel(resnet32(num_classes=num_class))
    model = model.to(device)
    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss().to(device)
    
    ####create z initialization#########
    z = np.zeros(num_train)
    
    gamma = 0.3
    eta = 0.01
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
        train_loss, train_acc = train(train_loader, model, criterion, optimizer)
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

    file_name = 'model.th'
    mdel_data = {"state_dict": model.state_dict()}
    torch.save(mdel_data, os.path.join(model_loc, file_name))

    results = utils.class_accuracy(val_loader, model, args)
    results["OA"] = val_acc
    hyper_param = utils.log_hyperparameter(args, args.tro_train)
    pprint(results)
    writer.add_hparams(hparam_dict=hyper_param, metric_dict=results)
    writer.close()

def q(model,x_i,y_i,x_j,y_j,gamma):
    output_i = model(x_i)
    loss_i = criterion(output_i, y_i)
    loss_i.backward()
    para = model.parameters()
    grad_i = para.grad
    grad_i.flatten()
    grad_i = grad_i/np.linalg.norm(grad_i)

    output_j = model(x_j)
    loss_j = criterion(output_j, y_j)
    loss_j.backward()
    para = model.parameters()
    grad_j = para.grad
    grad_j.flatten()
    grad_j = grad_j/np.linalg.norm(grad_j)


    return max( np.inner(grad_i, grad_j)-gamma ,0)



def train(train_dataset, model, criterion, optimizer):
    """ Run one train epoch """

    losses = utils.AverageMeter()
    accuracies = utils.AverageMeter()

    model.train()
    num_batches = num_train/args.batch_size
    arr = np.arange(num_train)
    arr1 = random.shuffle(arr)
    arr2 = random.shuffle(arr)
    for t in range(num_batches):
    # for _, (inputs, target) in enumerate(train_loader):
        B1, Y1= train_dataset[arr1[t]*batch_size:(arr1[t]+1)*batch_size]
        Y1 = Y1.to(device)
        print(Y1.shape)
        B1_var = B1.to(device)
        Y1_var = Y1
        print('minibatch size is:'+str(B1_var.shape))


        B2, Y2= train_dataset[arr2[t]*batch_size:(arr2[t]+1)*batch_size]
        Y2 = Y2.to(device)
        B2_var = B2.to(device)
        Y2_var = Y2
 
        output = model(input_var)
        acc = utils.accuracy(output.data, target)
        print('Output dimension is:'+str(output.shape))
        # if args.logit_adj_train:
        #     output = output + args.logit_adjustments

        loss_r = 0
        for parameter in model.parameters():
            loss_r += torch.sum(parameter ** 2)
        loss = loss + args.weight_decay * loss_r


         #####update z to approx exp of sum #######
        for i in range(arr1[t]*batch_size:(arr1[t]+1)*batch_size):
            x_i,y_i = train_dataset[i]
            corr = 0
            for j in range(arr2[t]*batch_size:(arr2[t]+1)*batch_size)
                x_j,y_j = train_dataset[j]
                corr = corr + q(model, x_i,y_i,x_j,y_j,gamma) - q(old_model, x_i,y_i,x_j,y_j,gamma) 
            z[i] = (1-beta)*(z[i]+corr1) + beta*corr1


        #####compute stochastic gradients#######

        old_model = model 
        weighted_grad = np.zeros(model.parameters().shape)
        
        for i in range(arr1[t]*batch_size:(arr1[t]+1)*batch_size):
            x_i,y_i = train_dataset[i]
            output_i = model(x_i)
            loss_i = criterion(output_i, y_i)
            loss_i.backward()
            para = model.parameters()
            grad_i = para.grad
            weighted_grad = weighted_grad + math.exp(-z[i])*grad_i 
 
        model.parameters() = model.parameters() - eta* weighted_grad
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))


    return losses.avg, accuracies.avg


def validate(val_loader, model, criterion):
    """ Run evaluation """

    losses = utils.AverageMeter()
    accuracies = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for _, (inputs, target) in enumerate(val_loader):
            target = target.to(device)
            input_var = inputs.to(device)
            target_var = target.to(device)

            output = model(input_var)
            loss = criterion(output, target_var)

            if args.logit_adj_post:
                output = output - args.logit_adjustments

            elif args.logit_adj_train:
                loss = criterion(output + args.logit_adjustments, target_var)

            acc = utils.accuracy(output.data, target)
            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))

    return losses.avg, accuracies.avg


if __name__ == '__main__':
    main()
