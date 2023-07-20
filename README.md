# FedMCSA: Personalized Federated Learning via Model Components Self-Attention
This is the official implementation of our paper **FedMCSA: Personalized Federated Learning via Model Components Self-Attention**.
This research project is developed based on Python 3 and Pytorch.

Authors: Qi Guo, Yong Qi, Saiyu Qi, Di Wu, and Qian Li

Full paper: https://arxiv.org/pdf/2208.10731.pdf

# Software requirements:
- numpy, scipy, torch, Pillow, matplotlib.

- To download the dependencies: **pip3 install -r requirements.txt**

# Code References During Implementation
  - [pFedMe](https://github.com/CharlieDinh/pFedMe)
  
# Dataset: We use 4 datasets: Synthetic, MNIST, Fashion-MNIST, and Cifar10.
- To generate non-idd Synthetic data:
  - Access data/Synthetic and run: "python3 generate_synthetic_05_05.py". 

- To generate non-iid MNIST Data: 
  - Access data/Mnist and run: "python3 generate_niid_20users.py"

- To generate non-iid Fashion-MNIST Data:
  - Access data/FMnist and run: "python3 generate_niid_20users_Fmnist.py"

- To generate non-iid Cifar10 Data:
  - Access data/Cifar10 and run: "python3 genenerate_niid_users_cifar10.py"

# Produce experiments:

- There is a main file "main.py" which allows running all experiments.
- To run the experiments, access the main file and run: "python3 main.py"
