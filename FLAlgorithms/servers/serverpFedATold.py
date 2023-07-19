import torch
import os

from FLAlgorithms.users.userpFedATold import UserpFedATold
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np
import copy
import h5py
# Implementation for pFedATold Server

class pFedATold(Server):
    def __init__(self, device,  dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                 local_epochs, optimizer, num_users, K, personal_learning_rate, times, at_hyper_c, at_num_of_iter, at_num_of_method, model_net, at_qkvsame):
        super().__init__(device, dataset,algorithm, model[0], batch_size, learning_rate, beta, lamda, num_glob_iters,
                         local_epochs, optimizer, num_users, times, model_net)

        # Initialize data for all  users
        data = read_data(dataset)
        total_users = len(data[0])
        self.K = K
        self.personal_learning_rate = personal_learning_rate
        # self.model_group = [copy.deepcopy(model) for i in range(total_users)]
        # New add
        self.at_hyper_c = at_hyper_c
        self.at_num_of_iter = at_num_of_iter
        self.at_num_of_method = at_num_of_method
        # self.model_net = model_net
        self.at_qkvsame = at_qkvsame

        # save attention corr
        self.attention_onetimes = []
        self.attention_oneglobal = []


        for i in range(total_users):
            id, train , test = read_user_data(i, data, dataset)
            user = UserpFedATold(device, id, train, test, model, batch_size, learning_rate, beta, lamda, local_epochs, optimizer, K, personal_learning_rate)
            self.users.append(user)
            self.total_train_samples += user.train_samples
        print("Number of users / total users:",num_users, " / " ,total_users)
        print("Finished creating pFedMe server.")

    def send_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for user in self.users:
            user.set_grads(grads)

    def train(self):
        loss = []
        self.attention_oneglobal = []
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")

            # send all parameter for users 
            # self.send_parameters()
            self.send_parameters_group()

            # Evaluate gloal model on user for each interation
            print("Evaluate global model")
            print("")
            self.evaluate()

            # do update for all users not only selected users
            for user in self.users:
                user.train(self.local_epochs) #* user.train_samples
            
            # choose several users to send back upated model to server
            # self.personalized_evaluate()
            self.selected_users = self.select_users(glob_iter,self.num_users)
            
            # Evaluate gloal model on user for each interation
            #print("Evaluate persionalized model")
            #print("")
            self.evaluate_personalized_model()
            #self.aggregate_parameters()
            self.persionalized_aggregate_parameters_attention()

            # save attention corr
            self.attention_oneglobal.append(self.attention_onetimes)
            
            # self.send_parameters()
            #
            # # Evaluate model each interation
            # self.evaluate()
            #
            # self.selected_users = self.select_users(glob_iter, self.num_users)
            # for user in self.selected_users:
            #     user.train(self.local_epochs)  # * user.train_samples
            # self.aggregate_parameters()


        # save attention corr
        alg_temp = "attentioncorr" + self.dataset + "_" + self.model_net + "_" + self.algorithm + "_" + str(self.learning_rate) + "_" + str(self.lamda) + "_" + str(self.at_hyper_c) + "_" + str(self.times)
        with h5py.File("./results/" + '{}.h5'.format(alg_temp), 'w') as hf:
            hf.create_dataset('attention_corr', data=self.attention_oneglobal)
            hf.close()


        #print(loss)
        self.save_results()
        self.save_model()
