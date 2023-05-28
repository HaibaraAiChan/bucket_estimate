import sys
sys.path.insert(0,'..')
sys.path.insert(0,'..')
sys.path.insert(0,'../../pytorch/utils/')
sys.path.insert(0,'../../pytorch/bucketing/')
sys.path.insert(0,'../../pytorch/models/')
import dgl
from dgl.data.utils import save_graphs
import numpy as np
from statistics import mean
import torch
import gc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

from bucketing_dataloader import generate_dataloader_bucket_block

import dgl.nn.pytorch as dglnn
import time
import argparse
import tqdm

import random
from graphsage_model_wo_mem import GraphSAGE
import dgl.function as fn
from load_graph import load_reddit, inductive_split, load_ogb, load_cora, load_karate, prepare_data, load_pubmed

from load_graph import load_ogbn_dataset
from memory_usage import see_memory_usage, nvidia_smi_usage
import tracemalloc
from cpu_mem_usage import get_memory
from statistics import mean

from my_utils import parse_results
from collections import Counter

import pickle
from utils import Logger
import os 
import numpy
import pdb



def set_seed(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.device >= 0:
		torch.cuda.manual_seed_all(args.seed)
		torch.cuda.manual_seed(args.seed)
		torch.backends.cudnn.enabled = False
		torch.backends.cudnn.deterministic = True
		dgl.seed(args.seed)
		dgl.random.seed(args.seed)

def CPU_DELTA_TIME(tic, str1):
	toc = time.time()
	print(str1 + ' spend:  {:.6f}'.format(toc - tic))
	return toc


def compute_acc(pred, labels):
	"""
	Compute the accuracy of prediction given the labels.
	"""
	labels = labels.long()
	return (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def evaluate(model, g, nfeats, labels, train_nid, val_nid, test_nid, device, args):
	"""
	Evaluate the model on the validation set specified by ``val_nid``.
	g : The entire graph.
	inputs : The features of all the nodes.
	labels : The labels of all the nodes.
	val_nid : the node Ids for validation.
	device : The GPU device to evaluate on.
	"""
	# train_nid = train_nid.to(device)
	# val_nid=val_nid.to(device)
	# test_nid=test_nid.to(device)
	nfeats=nfeats.to(device)
	g=g.to(device)
	# print('device ', device)
	model.eval()
	with torch.no_grad():
		# pred = model(g=g, x=nfeats)
		pred = model.inference(g, nfeats,  args, device)
	model.train()

	train_acc= compute_acc(pred[train_nid], labels[train_nid].to(pred.device))
	val_acc=compute_acc(pred[val_nid], labels[val_nid].to(pred.device))
	test_acc=compute_acc(pred[test_nid], labels[test_nid].to(pred.device))
	return (train_acc, val_acc, test_acc)


def load_subtensor(nfeat, labels, seeds, input_nodes, device):
	"""
	Extracts features and labels for a subset of nodes
	"""
	batch_inputs = nfeat[input_nodes].to(device)
	batch_labels = labels[seeds].to(device)
	return batch_inputs, batch_labels

def load_block_subtensor(nfeat, labels, blocks, device,args):
	"""
	Extracts features and labels for a subset of nodes
	"""

	# if args.GPUmem:
	# 	see_memory_usage("----------------------------------------before batch input features to device")
	batch_inputs = nfeat[blocks[0].srcdata[dgl.NID]].to(device)
	# if args.GPUmem:
	# 	see_memory_usage("----------------------------------------after batch input features to device")
	batch_labels = labels[blocks[-1].dstdata[dgl.NID]].to(device)
	# print('input global nids ', blocks[0].srcdata[dgl.NID])
	# print('input features: ', batch_inputs)
	# print('seeds global nids ', blocks[-1].dstdata[dgl.NID])
	# print('seeds labels : ',batch_labels)
	# if args.GPUmem:
	# 	see_memory_usage("----------------------------------------after  batch labels to device")
	return batch_inputs, batch_labels

def get_compute_num_nids(blocks):
	res=0
	for b in blocks:
		res+=len(b.srcdata['_ID'])
	return res


def get_FL_output_num_nids(blocks):

	output_fl =len(blocks[0].dstdata['_ID'])
	return output_fl


def knapsack_float(items, capacity):
    n = len(items)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        value, weight = items[i - 1]
        for j in range(capacity + 1):
            if weight <= j:
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][int(j - weight)] + value)
            else:
                dp[i][j] = dp[i - 1][j]
	# Find the optimal items
    optimal_items = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            optimal_items.append(i - 1)
            w -= items[i - 1][1]
            w = int(w)
    return dp[-1][-1], optimal_items


def EST_mem(modified_mem, optimal_items):
    # print(modified_mem)
    # print(optimal_items)
    result = 0
    for idx, ll in enumerate(modified_mem):
        if idx in optimal_items:
            result += ll[1]

    return result
    
    


def knapsack(items, capacity):
    n = len(items)
    # Initialize the dynamic programming table
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

    # Fill the table using dynamic programming
    for i in range(1, n + 1):
        item_value, item_weight = items[i - 1]
        for w in range(capacity + 1):
            if item_weight <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - item_weight] + item_value)
            else:
                dp[i][w] = dp[i - 1][w]

    # Find the optimal items
    optimal_items = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            optimal_items.append(i - 1)
            w -= items[i - 1][1]

    return dp[n][capacity], optimal_items

# # Example usage:
# items = [(60, 10), (100, 20), (120, 30)]  # (value, weight)
# capacity = 50
# max_value, optimal_items = knapsack(items, capacity)
# print("Maximum value:", max_value)
# print("Optimal items:", optimal_items)
def print_mem(list_mem):
    deg = 1
    for item in list_mem:
        print('degree '+str(deg) +' '+str(item[0]))
        deg += 1
    print()
    
def estimate_mem(data_dict, in_feat, hidden_size, redundant_ratio):	
	
	SUM_mem =0
	estimated_mem_list = []
	modified_estimated_mem_list = []
	# pdb.set_trace()


	weight = 1
	for i, data in enumerate(data_dict):
		estimated_mem = 0
		for i in range (len(data)):
			sum_b = 0
			for idx, (key, val) in enumerate(data[i].items()):
				sum_b = sum_b + key*val
				if idx ==0: # the input layer, in_feat 100
					estimated_mem  +=  sum_b*in_feat*18*4/1024/1024/1024
				if idx ==1: # the output layer
					estimated_mem  +=  sum_b*hidden_size*18*4/1024/1024/1024
				

		estimated_mem_list.append([estimated_mem, weight])
		modified_estimated_mem_list.append([estimated_mem/redundant_ratio[i], weight]) # redundant_ratio[i] is a variable depends on graph characteristic
		SUM_mem += estimated_mem
		# print('  estimated memory /GB degree '+str(ii)+': '+str(estimated_mem) )  
	print('sum mem of all batches directly ',SUM_mem)
	print('modified 1-24 degree bucket batches mem estimated :')
	print([m[0] for m in (modified_estimated_mem_list[:-1])])
	print('sum of them')
	print(sum([m[0] for m in (modified_estimated_mem_list[:-1])]))


	

	return modified_estimated_mem_list, estimated_mem_list

def estimate(data_dict, in_feat, hidden_size):
	weight = 1 #----------------------
	ii = 1
	all_mem =0
	estimated_mem_list = []
	modified_estimated_mem_list = []
	for data in data_dict:
		estimated_mem = 0
		for i in range (len(data)):
			sum_b = 0
			for  key, val in (data[i].items()):
				sum_b = sum_b + key*val
			mem_tmp = sum_b*128*18*4/1024/1024/1024
			# print(mem_tmp)
			estimated_mem  +=  mem_tmp
		estimated_mem_list.append([estimated_mem, weight])
		modified_estimated_mem_list.append([estimated_mem/3, weight]) # 3 is a variable depends on graph characteristic
		all_mem += estimated_mem
		# print('  estimated memory /GB degree '+str(ii)+': '+str(estimated_mem) )  
		ii+=1
	print(all_mem)

	return modified_estimated_mem_list, estimated_mem_list

#### Entry point
def run(args, device, data):
	if args.GPUmem:
		see_memory_usage("----------------------------------------start of run function ")
	# Unpack data
	g, nfeats, labels, n_classes, train_nid, val_nid, test_nid = data
	in_feats = len(nfeats[0])
	# print('in feats: ', in_feats)
	nvidia_smi_list=[]

	if args.selection_method =='metis':
		args.o_graph = dgl.node_subgraph(g, train_nid)


	sampler = dgl.dataloading.MultiLayerNeighborSampler(
		[int(fanout) for fanout in args.fan_out.split(',')])
	full_batch_size = len(train_nid)


	args.num_workers = 0


	model = GraphSAGE(
					in_feats,
					args.num_hidden,
					n_classes,
					args.aggre,
					args.num_layers,
					F.relu,
					args.dropout).to(device)

	loss_fcn = nn.CrossEntropyLoss()

	# if args.GPUmem:
	# 	see_memory_usage("----------------------------------------after model to device")
	logger = Logger(args.num_runs, args)
	for run in range(args.num_runs):
		model.reset_parameters()
		# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
		for epoch in range(args.num_epochs):
			model.train()

			loss_sum=0
			# start of data preprocessing part---s---------s--------s-------------s--------s------------s--------s----
			if args.load_full_batch:
				full_batch_dataloader=[]
				file_name=r'/home/cc/Betty_baseline/dataset/fan_out_'+args.fan_out+'/'+args.dataset+'_'+str(epoch)+'_items.pickle'
				with open(file_name, 'rb') as handle:
					item=pickle.load(handle)
					full_batch_dataloader.append(item)
			
			if args.num_batch > 1:
				b_block_dataloader, weights_list, time_collection = generate_dataloader_bucket_block(g, full_batch_dataloader, args)
				data_dict = []
				for step, (input_nodes, seeds, blocks) in enumerate(b_block_dataloader):
					layer = 0
					dict_list =[]
					for b in blocks:
						# print('layer ', layer)
						graph_in = dict(Counter(b.in_degrees().tolist()))
						graph_in = dict(sorted(graph_in.items()))

						print(graph_in)
						dict_list.append(graph_in)
						# print('dst nids of current layer ', len(b.dstdata['_ID']))
						layer = layer +1
					print()
					data_dict.append(dict_list)
				print('data_dict')
				print(data_dict)
				redundant_ratio =[]
				for step, (input_nodes, seeds, blocks) in enumerate(b_block_dataloader):
					print(len(input_nodes)/len(seeds)/(25*0.441))
				# redundant_ratio = [7.976878612716763, 7.248868778280543, 5.85204991087344, 5.722039473684211, 6.154157303370786, 5.834545454545455, 5.287677208287896, 6.475830078125, 6.010030864197531, 5.838125, 6.01923076923077, 5.901943967981704, 6.146466541588493, 5.490133607399794, 5.587518573551263, 5.859324104234528, 6.388806601777402, 5.797001471670345, 6.045331651045422, 5.7749125874125875, 5.94677393141557, 6.1750614250614255, 5.81568706387547, 5.67984496124031, 0.395462923578619]
				# redundant_ratio = [3.2784971098265894, 2.979285067873303, 2.4051925133689838, 2.3517582236842105, 2.529358651685393, 2.3979981818181817, 2.173235332606325, 2.661566162109375, 2.470122685185185, 2.399469375, 2.4739038461538465, 2.42569897084048, 2.5261977485928706, 2.256444912641315, 2.296470133729569, 2.4081822068403906, 2.625799513330512, 2.382567604856512, 2.4846313085796683, 2.3734890734265734, 2.444124085811799, 2.537950245700246, 2.390247383252818, 2.3344162790697673, 0.1625352615908124]
				# redundant_ratio = [3.5178034682080925, 3.1967511312217196, 2.580754010695187, 2.5234194078947367, 2.7139833707865164, 2.573034545454546, 2.331865648854962, 2.855841064453125, 2.650423611111111, 2.574613125, 2.6544807692307697, 2.6027572898799316, 2.7105917448405257, 2.4211489208633092, 2.464095690936107, 2.5839619299674266, 2.817463711383834, 2.5564776490066223, 2.665991258111031, 2.546736451048951, 2.6225273037542665, 2.723202088452089, 2.564717995169082, 2.5048116279069768, 0.174399149298171]
				# redundant_ratio = [1.445817225433526, 1.3138647149321268, 1.0606898983957218, 1.0371253766447368, 1.1154471653932583, 1.0575171981818183, 0.9583967816793892, 1.1737506774902342, 1.0893241041666666, 1.058165994375, 1.0909915961538463, 1.069733246140652, 1.114053207129456, 0.99509220647482, 1.0127433289747398, 1.0620083532166122, 1.1579775853787557, 1.0507123137417218, 1.0957224070836338, 1.0467086813811188, 1.0778587218430034, 1.1192360583538086, 1.0540990960144927, 1.0294775790697674, 0.07167805036154827]
				# redundant_ratio = [1.7414868654495508, 1.5825500649612474, 1.277600995393657, 1.249217528660761, 1.3435561241517409, 1.2737794779477947, 1.1543889350767138, 1.4137827051748144, 1.312090896589659, 1.274560952970297, 1.3140993907083016, 1.2884937078613523, 1.3418771014062008, 1.1985885746848064, 1.2198493519485678, 1.2791890742413004, 1.3947840155365516, 1.2655829945577337, 1.3197976525302135, 1.260760619331164, 1.2982808434427062, 1.3481198457683607, 1.2696623738460804, 1.2400057563895923, 0.08633621252384703]
				# pdb.set_trace()
				modified_res, res = estimate_mem(data_dict, 100, args.num_hidden, redundant_ratio)
				# items = [(60, 10), (100, 20), (120, 30)]  # (value, weight)
				# items = [( mem_estimation_1, 1), ( mem_estimation_2, 1), (mem_estimation_3,  1)...]  # (value, weight)
				# 
				# items = res
				print('the memory estimated for each bucket batch is shown below: ')
				print_mem(res)
				print()
				print('the modified memory estimated for each bucket batch is shown below: ')
				print_mem(modified_res)
				print()
				# items = modified_res
				# capacity = 22
				# max_value, optimal_items = knapsack_float(items, capacity)
				# # max_value, optimal_items = knapsack(items, capacity)
				# print("Maximum value:", max_value)
				# print("Optimal items:", optimal_items)
				# est_mem = EST_mem(modified_res, optimal_items)
				# print('the estimated memory under this constraint ', est_mem)


			elif args.num_batch == 1:
				# print('orignal labels: ', labels)
				for step, (input_nodes, seeds, blocks) in enumerate(full_batch_dataloader):
					# print()
					print('full batch src global ', len(input_nodes))
					print('full batch dst global ', len(seeds))
					# print('full batch eid global ', blocks[-1].edata['_ID'])
					batch_inputs, batch_labels = load_block_subtensor(nfeats, labels, blocks, device,args)#------------*
					# print('batch_labels ')
					# print(batch_labels)
					# print('blocks')
					# print(blocks[0].edata['_ID'])
					# print(blocks[-1].edata['_ID'])
					see_memory_usage("----------------------------------------after load_block_subtensor")
					blocks = [block.int().to(device) for block in blocks]
					see_memory_usage("----------------------------------------after block to device")

					batch_pred = model(blocks, batch_inputs)
					see_memory_usage("----------------------------------------after model")

					loss = loss_fcn(batch_pred, batch_labels)
					print('full batch train ------ loss ' + str(loss.item()) )
					see_memory_usage("----------------------------------------after loss")

					loss.backward()
					see_memory_usage("----------------------------------------after loss backward")

					optimizer.step()
					optimizer.zero_grad()
					print()
					see_memory_usage("----------------------------------------full batch")
					

def main():
	# get_memory("-----------------------------------------main_start***************************")
	tt = time.time()
	print("main start at this time " + str(tt))
	argparser = argparse.ArgumentParser("multi-gpu training")
	argparser.add_argument('--device', type=int, default=0,
		help="GPU device ID. Use -1 for CPU training")
	argparser.add_argument('--seed', type=int, default=1236)
	argparser.add_argument('--setseed', type=bool, default=True)
	argparser.add_argument('--GPUmem', type=bool, default=True)
	argparser.add_argument('--load-full-batch', type=bool, default=True)
	# argparser.add_argument('--root', type=str, default='../my_full_graph/')
	# argparser.add_argument('--dataset', type=str, default='ogbn-arxiv')
	# argparser.add_argument('--dataset', type=str, default='ogbn-mag')
	argparser.add_argument('--dataset', type=str, default='ogbn-products')
	# argparser.add_argument('--dataset', type=str, default='cora')
	# argparser.add_argument('--dataset', type=str, default='karate')
	# argparser.add_argument('--dataset', type=str, default='reddit')
	# argparser.add_argument('--aggre', type=str, default='mean')
	argparser.add_argument('--aggre', type=str, default='lstm')

	argparser.add_argument('--selection-method', type=str, default='range_bucketing')
	# argparser.add_argument('--selection-method', type=str, default='random_bucketing')
	# argparser.add_argument('--selection-method', type=str, default='fanout_bucketing')
	# argparser.add_argument('--selection-method', type=str, default='custom_bucketing')
	argparser.add_argument('--num-batch', type=int, default=10)

	argparser.add_argument('--num-runs', type=int, default=1)
	argparser.add_argument('--num-epochs', type=int, default=1)

	# argparser.add_argument('--num-hidden', type=int, default=128)
	argparser.add_argument('--num-hidden', type=int, default=512)

	# argparser.add_argument('--num-layers', type=int, default=1)
	# argparser.add_argument('--fan-out', type=str, default='10')

	argparser.add_argument('--num-layers', type=int, default=2)
	argparser.add_argument('--fan-out', type=str, default='10,25')
	# argparser.add_argument('--num-layers', type=int, default=3)
	# argparser.add_argument('--fan-out', type=str, default='10,25,30')



	# argparser.add_argument('--num-layers', type=int, default=1)
	# argparser.add_argument('--fan-out', type=str, default='4')
	# argparser.add_argument('--num-layers', type=int, default=2)
	# argparser.add_argument('--fan-out', type=str, default='2,4')


	argparser.add_argument('--log-indent', type=float, default=0)
#--------------------------------------------------------------------------------------


	argparser.add_argument('--lr', type=float, default=1e-2)
	argparser.add_argument('--dropout', type=float, default=0.5)
	argparser.add_argument("--weight-decay", type=float, default=5e-4,
						help="Weight for L2 loss")
	argparser.add_argument("--eval", action='store_true', 
						help='If not set, we will only do the training part.')

	argparser.add_argument('--num-workers', type=int, default=4,
		help="Number of sampling processes. Use 0 for no extra process.")
	

	args = argparser.parse_args()
	if args.setseed:
		set_seed(args)
	device = "cpu"
	if args.GPUmem:
		see_memory_usage("-----------------------------------------before load data ")
	if args.dataset=='karate':
		g, n_classes = load_karate()
		print('#nodes:', g.number_of_nodes())
		print('#edges:', g.number_of_edges())
		print('#classes:', n_classes)
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='cora':
		g, n_classes = load_cora()
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='pubmed':
		g, n_classes = load_pubmed()
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='reddit':
		g, n_classes = load_reddit()
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
		print('#nodes:', g.number_of_nodes())
		print('#edges:', g.number_of_edges())
		print('#classes:', n_classes)
	elif args.dataset == 'ogbn-arxiv':
		data = load_ogbn_dataset(args.dataset,  args)
		device = "cuda:0"

	elif args.dataset=='ogbn-products':
		g, n_classes = load_ogb(args.dataset,args)
		print('#nodes:', g.number_of_nodes())
		print('#edges:', g.number_of_edges())
		print('#classes:', n_classes)
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='ogbn-mag':
		# data = prepare_data_mag(device, args)
		data = load_ogbn_mag(args)
		device = "cuda:0"
		# run_mag(args, device, data)
		# return
	else:
		raise Exception('unknown dataset')


	best_test = run(args, device, data)


if __name__=='__main__':
	main()
