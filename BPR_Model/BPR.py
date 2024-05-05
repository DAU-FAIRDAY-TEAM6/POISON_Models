import numpy as np 
import pandas as pd 
import scipy.sparse as sp

import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn

#https://github.com/guoyang9/BPR-pytorch

def load_all(test_num=100):
    # 경로 설정
    path = './dataset/'
    
    # 훈련 데이터 로드: user와 item 컬럼만 사용
    train_data = pd.read_csv(
        path + 'train.csv', 
        sep=',', header=None, names=['user', 'item'], 
        usecols=[0, 1], dtype={0: np.int32, 1: np.int32})

    # 최대 user ID와 item ID를 계산하여 전체 사용자 수와 아이템 수를 파악
    user_num = train_data['user'].max() + 1
    item_num = train_data['item'].max() + 1

    # 리스트 형태로 데이터를 변환
    train_data = train_data.values.tolist()

    # 훈련 데이터를 희소 행렬 형태로 변환: user-item 행렬
    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for x in train_data:
        train_mat[x[0], x[1]] = 1.0  # 사용자-아이템 간 상호작용을 1로 설정

    # 테스트 데이터 로드
    test_data = []
    with open(path + 'test_neg_100.csv', 'r') as fd:
        line = fd.readline()
        while line != None and line != '':
            arr = line.split('\t')
            u = eval(arr[0])[0]  # 사용자 ID
            test_data.append([u, eval(arr[0])[1]])  # 긍정적 아이템 샘플
            for i in arr[1:]:
                test_data.append([u, int(i)])  # 부정적 아이템 샘플 추가
            line = fd.readline()

    return train_data, test_data, user_num, item_num, train_mat

# BPR 데이터셋 클래스: PyTorch의 Dataset을 상속받아 사용자 정의 데이터셋 생성
class BPRData(data.Dataset):
    def __init__(self, features, num_item, train_mat=None, num_ng=0, is_training=None):
        super(BPRData, self).__init__()
        self.features = features  # 원본 데이터
        self.num_item = num_item  # 전체 아이템 수
        self.train_mat = train_mat  # 훈련 데이터의 희소 행렬
        self.num_ng = num_ng  # 부정적 샘플의 수
        self.is_training = is_training  # 훈련 모드 여부

    # 부정적 샘플링 수행 함수
    def ng_sample(self):
        assert self.is_training, 'no need to sampling when testing'
        self.features_fill = []
        for x in self.features:
            u, i = x[0], x[1]
            for t in range(self.num_ng):
                j = np.random.randint(self.num_item)
                while (u, j) in self.train_mat:
                    j = np.random.randint(self.num_item)
                self.features_fill.append([u, i, j])  # 사용자, 긍정적 아이템, 부정적 아이템

    # 데이터셋 크기 반환
    def __len__(self):
        return self.num_ng * len(self.features) if self.is_training else len(self.features)

    # 데이터셋 요소 접근 함수
    def __getitem__(self, idx):
        features = self.features_fill if self.is_training else self.features
        user = features[idx][0]
        item_i = features[idx][1]
        item_j = features[idx][2] if self.is_training else features[idx][1]
        return user, item_i, item_j  # 사용자, 긍정적 아이템, 부정적 아이템 반환

# BPR 모델 정의 클래스: PyTorch의 Module을 상속받아 모델 정의
class BPR(nn.Module):
    def __init__(self, user_num, item_num, factor_num):
        super(BPR, self).__init__()
        self.embed_user = nn.Embedding(user_num, factor_num)  # 사용자 임베딩
        self.embed_item = nn.Embedding(item_num, factor_num)  # 아이템 임베딩

        # 임베딩 가중치 초기화
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

    # 순전파 함수
    def forward(self, user, item_i, item_j):
        user = self.embed_user(user)
        item_i = self.embed_item(item_i)
        item_j = self.embed_item(item_j)
        prediction_i = (user * item_i).sum(dim=-1)  # 긍정적 예측
        prediction_j = (user * item_j).sum(dim=-1)  # 부정적 예측
        return prediction_i, prediction_j





if __name__ == '__main__':
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
		help="sample negative itemes for training")
	parser.add_argument("--test_num_ng", 
		type=int,
		default=99, 
		help="sample part of negative itemes for testing")
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
	train_data, test_data, user_num ,item_num, train_mat = load_all()

	# construct the train and test datasets
	train_dataset = BPRData(
			train_data, item_num, train_mat, args.num_ng, True)
	test_dataset = BPRData(
			test_data, item_num, train_mat, 0, False)
	train_loader = data.DataLoader(train_dataset,
			batch_size=args.batch_size, shuffle=True, num_workers=4)
	test_loader = data.DataLoader(test_dataset,
			batch_size=args.test_num_ng+1, shuffle=False, num_workers=0)

	########################### CREATE MODEL #################################
	model = BPR(user_num, item_num, args.factor_num)
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

		for user, item_i, item_j in train_loader:
			user = user.cuda()
			item_i = item_i.cuda()
			item_j = item_j.cuda()

			model.zero_grad()
			prediction_i, prediction_j = model(user, item_i, item_j)
			loss = - (prediction_i - prediction_j).sigmoid().log().sum()
			loss.backward()
			optimizer.step()
			# writer.add_scalar('data\\loss', loss.item(), count)
			count += 1

		model.eval()
		HR, NDCG = metrics(model, test_loader, args.top_k)

		elapsed_time = time.time() - start_time
		print("The time elapse of epoch {:03d}".format(epoch) + " is: " + 
				time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
		print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))

		# if HR > best_hr:
		# 	best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
		# 	if args.out:
		# 		if not os.path.exists(path + model_path):
		# 			os.mkdir(path + model_path)
		# 		torch.save(model, '{}BPR.pt'.format(path + model_path))

	# print("End. Best epoch {:03d}: HR = {:.3f}, \
	# 	NDCG = {:.3f}".format(best_epoch, best_hr, best_ndcg))