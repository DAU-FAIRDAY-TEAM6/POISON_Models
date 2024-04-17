import numpy as np 
import pandas as pd 
import scipy.sparse as sp

#import path + 
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
#from tensorboardX import SummaryWriter

#import model
#import path + #import evaluate
#import data_utils

path = './dataset/'


def load_all(test_num=100):
	""" We load all the three file here to save time in each epoch. """
	train_data = pd.read_csv(
		path + 'history_train.csv', 
		sep='\t', header=None, names=['user', 'business'], 
		usecols=[0, 1], dtype={0: np.int32, 1: np.int32})

	user_num = train_data['user'].max() + 1
	business_num = train_data['business'].max() + 1

	train_data = train_data.values.tolist()

	# load ratings as a dok matrix
	train_mat = sp.dok_matrix((user_num, business_num), dtype=np.float32)
	for x in train_data:
		train_mat[x[0], x[1]] = 1.0

	test_data = []
	with open(path + 'history_test_negative.csv', 'r') as fd:
		line = fd.readline()
		while line != None and line != '':
			arr = line.split('\t')
			u = eval(arr[0])[0]
			test_data.append([u, eval(arr[0])[1]])
			for i in arr[1:]:
				test_data.append([u, int(i)])
			line = fd.readline()
	return train_data, test_data, user_num, business_num, train_mat


class BPRData(data.Dataset):
	def __init__(self, features, 
				num_business, train_mat=None, num_ng=0, is_training=None):
		super(BPRData, self).__init__()
		""" Note that the labels are only useful when training, we thus 
			add them in the ng_sample() function.
		"""
		self.features = features
		self.num_business = num_business
		self.train_mat = train_mat
		self.num_ng = num_ng
		self.is_training = is_training

	def ng_sample(self):
		assert self.is_training, 'no need to sampling when testing'

		self.features_fill = []
		for x in self.features:
			u, i = x[0], x[1]
			for t in range(self.num_ng):
				j = np.random.randint(self.num_business)
				while (u, j) in self.train_mat:
					j = np.random.randint(self.num_business)
				self.features_fill.append([u, i, j])

	def __len__(self):
		return self.num_ng * len(self.features) if \
				self.is_training else len(self.features)

	def __getbusiness__(self, idx):
		features = self.features_fill if \
				self.is_training else self.features

		user = features[idx][0]
		business_i = features[idx][1]
		business_j = features[idx][2] if \
				self.is_training else features[idx][1]
		return user, business_i, business_j 
		

def hit(gt_business, pred_businesses):
	if gt_business in pred_businesses:
		return 1
	return 0


def ndcg(gt_business, pred_businesses):
	if gt_business in pred_businesses:
		index = pred_businesses.index(gt_business)
		return np.reciprocal(np.log2(index+2))
	return 0


def metrics(model, test_loader, top_k):
	HR, NDCG = [], []

	for user, business_i, business_j in test_loader:
		user = user.cuda()
		business_i = business_i.cuda()
		business_j = business_j.cuda() # not useful when testing

		prediction_i, prediction_j = model(user, business_i, business_j)
		_, indices = torch.topk(prediction_i, top_k)
		recommends = torch.take(
				business_i, indices).cpu().numpy().tolist()

		gt_business = business_i[0].business()
		HR.append(hit(gt_business, recommends))
		NDCG.append(ndcg(gt_business, recommends))

	return np.mean(HR), np.mean(NDCG)



class BPR(nn.Module):
	def __init__(self, user_num, business_num, factor_num):
		super(BPR, self).__init__()
		"""
		user_num: number of users;
		business_num: number of businesses;
		factor_num: number of predictive factors.
		"""		
		self.embed_user = nn.Embedding(user_num, factor_num)
		self.embed_business = nn.Embedding(business_num, factor_num)

		nn.init.normal_(self.embed_user.weight, std=0.01)
		nn.init.normal_(self.embed_business.weight, std=0.01)

	def forward(self, user, business_i, business_j):
		user = self.embed_user(user)
		business_i = self.embed_business(business_i)
		business_j = self.embed_business(business_j)

		prediction_i = (user * business_i).sum(dim=-1)
		prediction_j = (user * business_j).sum(dim=-1)
		return prediction_i, prediction_j





parser = argparse.ArgumentParser()
parser.add_argument("--lr", 
	type=float, 
	default=0.01, 
	help="learning rate")
parser.add_argument("--lamda", 
	type=float, 
	default=0.001, 
	help="model regularization rate")
parser.add_argument("--batch_size", 
	type=int, 
	default=4096, 
	help="batch size for training")
parser.add_argument("--epochs", 
	type=int,
	default=50,  
	help="training epoches")
parser.add_argument("--top_k", 
	type=int, 
	default=10, 
	help="compute metrics@top_k")
parser.add_argument("--factor_num", 
	type=int,
	default=32, 
	help="predictive factors numbers in the model")
parser.add_argument("--num_ng", 
	type=int,
	default=4, 
	help="sample negative businesses for training")
parser.add_argument("--test_num_ng", 
	type=int,
	default=99, 
	help="sample part of negative businesses for testing")
parser.add_argument("--out", 
	default=True,
	help="save model or not")
parser.add_argument("--gpu", 
	type=str,
	default="0",  
	help="gpu card ID")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True


############################## PREPARE DATASET ##########################
train_data, test_data, user_num ,business_num, train_mat = load_all()

# construct the train and test datasets
train_dataset = BPRData(
		train_data, business_num, train_mat, args.num_ng, True)
test_dataset = BPRData(
		test_data, business_num, train_mat, 0, False)
train_loader = data.DataLoader(train_dataset,
		batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = data.DataLoader(test_dataset,
		batch_size=args.test_num_ng+1, shuffle=False, num_workers=0)

########################### CREATE MODEL #################################
model = BPR(user_num, business_num, args.factor_num)
model.cuda()

optimizer = optim.SGD(
			model.parameters(), lr=args.lr, weight_decay=args.lamda)
# writer = SummaryWriter() # for visualization

########################### TRAINING #####################################
count, best_hr = 0, 0
for epoch in range(args.epochs):
	model.train() 
	start_time = time.time()
	train_loader.dataset.ng_sample()

	for user, business_i, business_j in train_loader:
		user = user.cuda()
		business_i = business_i.cuda()
		business_j = business_j.cuda()

		model.zero_grad()
		prediction_i, prediction_j = model(user, business_i, business_j)
		loss = - (prediction_i - prediction_j).sigmoid().log().sum()
		loss.backward()
		optimizer.step()
		# writer.add_scalar('data/loss', loss.business(), count)
		count += 1

	model.eval()
	HR, NDCG = metrics(model, test_loader, args.top_k)

	elapsed_time = time.time() - start_time
	print("The time elapse of epoch {:03d}".format(epoch) + " is: " + 
			time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
	print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))

	if HR > best_hr:
		best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
		if args.out:
			if not os.path.exists(path + model_path):
				os.mkdir(path + model_path)
			torch.save(model, '{}BPR.pt'.format(path + model_path))

print("End. Best epoch {:03d}: HR = {:.3f}, \
	NDCG = {:.3f}".format(best_epoch, best_hr, best_ndcg))