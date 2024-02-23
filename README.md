# Peer-to-peer Federated Aggregation
This repository contains two implementations for two modules in a peer-to-peer federated learning setup. One is P2PK-SMOTE [1], a data rebalance aiming to rebalance the private datasets of clients and address the non-IID problem in federated learning. Another is SparSFA [2], a communication-efficient and robust aggregation rule which can be used to defend against poisoning attacks.  


Please cite: \
[1] Han Wang, Luis Muñoz-González, David Eklund, and Shahid Raza. 2021. Non-IID data re-balancing at IoT edge with peer-to-peer federated learning for anomaly detection. In Proceedings of the 14th ACM Conference on Security and Privacy in Wireless and Mobile Networks (WiSec '21). Association for Computing Machinery, New York, NY, USA, 153–163. https://doi.org/10.1145/3448300.3467827 \
[2] Han Wang, Luis Muñoz-González, Muhammad Zaid Hameed, David Eklund, Shahid Raza, SparSFA: Towards robust and communication-efficient peer-to-peer federated learning, Computers & Security, Volume 129, 2023, 103182. https://doi.org/10.1016/j.cose.2023.103182



## Use the code:
`$ python main.py` \
Arguments:
*	Train_dataset: string, mandatory: The dataset that is going to be executed.
*	-–balanced, Boolean, optional: It decides whether the data is split class-wise evenly for every client.
*	–fiveclient, Boolean, optional: The data is distributed to 5 clients for default experiment
*	–rebalancer, integer, mandatory: It is mandatory argument. If given 1, use the Data Rebalancer only for experiment
  *	1: Use the Data Rebalancer only
  *	Else: SparsFA
*	–attack_mode, integer, optional: Choose different types of attack.
  *	0: No attack is performed.
  *	1: Label Flipping attack.
  *	2: Data Noise attack.
  *	3: Objective Function Poisoning attack
  *	4: Byzantine attack  
*	–num_ads, integer, optional: Provide the number of the adversaries.
*	--random_network, Boolean, optional: Generate a random network
*	–num_client, integer, optional: If the random_network is specified, please provide the number of the clients joining the network.
