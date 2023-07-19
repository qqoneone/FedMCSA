#!/usr/bin/env python
import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import random
import os
from FLAlgorithms.servers.serveravg import FedAvg
from FLAlgorithms.servers.serverpFedATold import pFedATold
from FLAlgorithms.trainmodel.models import *
from utils.plot_utils import *
import torch
import copy

torch.manual_seed(0)

def main(dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
         local_epochs, optimizer, numusers, K, personal_learning_rate, times, gpu, at_hyper_c, at_num_of_iter, at_num_of_method, at_qkvsame):

    # Get device status: Check GPU or CPU
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")

    model_net = copy.deepcopy(model)

    for i in range(times):
        print("---------------Running time:------------",i)
        # Generate model
        if(model == "mclr"):
            if(dataset == "Mnist"):
                model = Mclr_Logistic().to(device), model
            elif(dataset == "Synthetic"):
                model = Mclr_Logistic(60,10).to(device), model
            elif (dataset == "Cifar100"):
                model = Mclr_Logistic(3072, 100).to(device), model
            elif (dataset == "Cifar10"):
                model = Mclr_Logistic(3072, 10).to(device), model
            else:
                model = Mclr_Logistic().to(device), model
                
        if(model == "cnn"):
            if(dataset == "Mnist"):
                model = CNN().to(device), model
            elif(dataset == "Cifar10"):
                model = CNNCifar(10).to(device), model
            
        if(model == "dnn"):
            if(dataset == "Mnist"):
                model = DNN().to(device), model
            elif (dataset == "Synthetic"):
                model = DNN(60,20,10).to(device), model
            elif (dataset == "Cifar100"):
                model = DNN(3072, 500, 100).to(device), model
            elif (dataset == "Cifar10"):
                model = DNN(3072, 100, 10).to(device), model
            else:
                model = DNN().to(device), model

        # select algorithm
        if(algorithm == "FedAvg"):
            server = FedAvg(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, numusers, i, model_net)

        
        if(algorithm == "pFedATold"):
            # server = pFedMe(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, numusers, K, personal_learning_rate, i)
            server = pFedATold(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, numusers, K, personal_learning_rate, i, at_hyper_c, at_num_of_iter, at_num_of_method, model_net, at_qkvsame)


        server.train()
        server.test()

    # Average data 
    if(algorithm == "pFedATold"):
        # average_data(num_users=numusers, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=lamda,learning_rate=learning_rate, beta = beta, algorithms="pFedMe_p", batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate,times = times)
        average_data(num_users=numusers, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=lamda,
                     learning_rate=learning_rate, beta=beta, algorithms="pFedATold_p", batch_size=batch_size,
                     dataset=dataset, k=K, personal_learning_rate=personal_learning_rate, times=times, model_net=model_net, at_hyper_c=at_hyper_c, at_num_of_iter=at_num_of_iter, at_num_of_method=at_num_of_method, at_qkvsame=at_qkvsame)

    # average_data(num_users=numusers, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=lamda,learning_rate=learning_rate, beta = beta, algorithms=algorithm, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate,times = times)
    average_data(num_users=numusers, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=lamda,
                 learning_rate=learning_rate, beta=beta, algorithms=algorithm, batch_size=batch_size, dataset=dataset,
                 k=K, personal_learning_rate=personal_learning_rate, times=times, model_net=model_net, at_hyper_c=at_hyper_c, at_num_of_iter=at_num_of_iter, at_num_of_method=at_num_of_method, at_qkvsame=at_qkvsame)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Mnist", choices=["Mnist", "Synthetic", "Cifar10", "FMnist"])
    parser.add_argument("--model", type=str, default="mclr", choices=["dnn", "mclr", "cnn"])
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.005, help="Local learning rate")
    parser.add_argument("--beta", type=float, default=1.0, help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
    parser.add_argument("--lamda", type=int, default=15, help="Regularization term")
    parser.add_argument("--num_global_iters", type=int, default=5)
    parser.add_argument("--local_epochs", type=int, default=20)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--algorithm", type=str, default="PerAvg", choices=["pFedMe", "PerAvg", "FedAvg", "FedAvgAdd", "pFedAT", "pFedATold","pFedAMP", "pFedAMPold"])
    parser.add_argument("--numusers", type=int, default=20, help="Number of Users per round")
    parser.add_argument("--K", type=int, default=5, help="Computation steps")
    parser.add_argument("--personal_learning_rate", type=float, default=0.01, help="Persionalized learning rate to caculate theta aproximately using K steps")
    parser.add_argument("--times", type=int, default=2, help="running time")
    parser.add_argument("--gpu", type=int, default=2, help="Which GPU to run the experiments, -1 mean CPU, 0,1,2 for GPU")
    parser.add_argument("--at_hyper_c", type=int, default=1000, help="hyper_c in attentions_simple, 2000 for Mnist, 450 for Synthetic")
    parser.add_argument("--at_num_of_iter", type=int, default=1, help="num_of_iter in attentions_simple")
    parser.add_argument("--at_num_of_method", type=int, default=2, help="num_of_method in attentions_simple")
    parser.add_argument("--at_qkvsame", type=int, default=0, help="flag for qkv same")
    args = parser.parse_args()

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("Learing rate       : {}".format(args.learning_rate))
    print("Average Moving       : {}".format(args.beta))
    print("Subset of users      : {}".format(args.numusers))
    print("Number of global rounds       : {}".format(args.num_global_iters))
    print("Number of local rounds       : {}".format(args.local_epochs))
    print("Dataset       : {}".format(args.dataset))
    print("Local Model       : {}".format(args.model))
    print("=" * 80)

    main(
        dataset=args.dataset,
        algorithm = args.algorithm,
        model=args.model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta = args.beta, 
        lamda = args.lamda,
        num_glob_iters=args.num_global_iters,
        local_epochs=args.local_epochs,
        optimizer= args.optimizer,
        numusers = args.numusers,
        K=args.K,
        personal_learning_rate=args.personal_learning_rate,
        times = args.times,
        gpu=args.gpu,
        at_hyper_c = args.at_hyper_c,
        at_num_of_iter=args.at_num_of_iter,
        at_num_of_method=args.at_num_of_method,
        at_qkvsame = args.at_qkvsame
        )
