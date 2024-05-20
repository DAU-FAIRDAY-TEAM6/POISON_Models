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

def load_all(test_num=100):
    path = './BPR_Model/dataset/'
    
    train_data = pd.read_csv(
        path + 'train.csv', 
        sep='\t', header=None, names=['user', 'item'], 
        usecols=[0, 1], dtype={0: np.int32, 1: np.int32})

    poi_data = pd.read_csv(
        path + 'poi.csv', 
        sep=',', header=None, names=['item', 'lat', 'lon'], 
        usecols=[0, 1, 2], dtype={0: np.int32, 1: np.float32, 2: np.float32})

    user_num = train_data['user'].max() + 1
    item_num = train_data['item'].max() + 1

    train_data = train_data.values.tolist()
    poi_data = poi_data.set_index('item').values

    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for x in train_data:
        train_mat[x[0], x[1]] = 1.0

    test_data = []
    with open(path + 'test_neg_100.csv', 'r') as fd:
        line = fd.readline()
        while line != None and line != '':
            arr = line.split('\t')
            u = eval(arr[0])[0]
            test_data.append([u, eval(arr[0])[1]])
            for i in arr[1:]:
                test_data.append([u, int(i)])
            line = fd.readline()

    return train_data, test_data, user_num, item_num, train_mat, poi_data

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = R * c
    return d

def calculate_distances(poi_data):
    item_num = poi_data.shape[0]
    distances = np.zeros((item_num, item_num))
    for i in range(item_num):
        for j in range(item_num):
            distances[i, j] = haversine(poi_data[i, 0], poi_data[i, 1], poi_data[j, 0], poi_data[j, 1])
    return distances

def normalize_distances(distances):
    min_d = np.min(distances)
    max_d = np.max(distances)
    norm_distances = 0.5 * (distances - min_d) / (max_d - min_d) + 0.5
    return norm_distances

class BPRData(data.Dataset):
    def __init__(self, features, num_item, train_mat=None, num_ng=0, is_training=None):
        super(BPRData, self).__init__()
        self.features = features
        self.num_item = num_item
        self.train_mat = train_mat
        self.num_ng = num_ng
        self.is_training = is_training

    def ng_sample(self):
        assert self.is_training, 'no need to sampling when testing'
        self.features_fill = []
        for x in self.features:
            u, i = x[0], x[1]
            for t in range(self.num_ng):
                j = np.random.randint(self.num_item)
                while (u, j) in self.train_mat:
                    j = np.random.randint(self.num_item)
                self.features_fill.append([u, i, j])

    def __len__(self):
        return self.num_ng * len(self.features) if self.is_training else len(self.features)

    def __getitem__(self, idx):
        features = self.features_fill if self.is_training else self.features
        user = features[idx][0]
        item_i = features[idx][1]
        item_j = features[idx][2] if self.is_training else features[idx][1]
        return user, item_i, item_j

class BPR(nn.Module):
    def __init__(self, user_num, item_num, factor_num):
        super(BPR, self).__init__()
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

    def forward(self, user, item_i, item_j, distance_ij):
        user = self.embed_user(user)
        item_i = self.embed_item(item_i)
        item_j = self.embed_item(item_j)
        prediction_i = (user * item_i).sum(dim=-1)
        prediction_j = (user * item_j).sum(dim=-1)
        return prediction_i, prediction_j, distance_ij
    
def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	return 0


def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return np.reciprocal(np.log2(index+2))
	return 0


# def metrics(model, test_loader, top_k):
# 	HR, NDCG = [], []

# 	for user, item_i, item_j in test_loader:
# 		user = user.cuda()
# 		item_i = item_i.cuda()
# 		item_j = item_j.cuda() # not useful when testing

# 		prediction_i, prediction_j = model(user, item_i, item_j)
# 		_, indices = torch.topk(prediction_i, top_k)
# 		recommends = torch.take(
# 				item_i, indices).cpu().numpy().tolist()

# 		gt_item = item_i[0].item()
# 		HR.append(hit(gt_item, recommends))
# 		NDCG.append(ndcg(gt_item, recommends))

# 	return np.mean(HR), np.mean(NDCG)

def metrics(model, test_loader, top_k):
	HR, NDCG = [], []

	for user, item_i, item_j in test_loader:
		user = user
		item_i = item_i
		item_j = item_j# not useful when testing

		prediction_i, prediction_j = model(user, item_i, item_j)
		_, indices = torch.topk(prediction_i, top_k)
		recommends = torch.take(
				item_i, indices).numpy().tolist()

		gt_item = item_i[0].item()
		HR.append(hit(gt_item, recommends))
		NDCG.append(ndcg(gt_item, recommends))

	return np.mean(HR), np.mean(NDCG)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--lamda", type=float, default=0.001, help="model regularization rate")
    parser.add_argument("--batch_size", type=int, default=4096, help="batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="training epoches")
    parser.add_argument("--top_k", type=int, default=10, help="compute metrics@top_k")
    parser.add_argument("--factor_num", type=int, default=32, help="predictive factors numbers in the model")
    parser.add_argument("--num_ng", type=int, default=4, help="sample negative itemes for training")
    parser.add_argument("--test_num_ng", type=int, default=99, help="sample part of negative itemes for testing")
    parser.add_argument("--out", default=True, help="save model or not")
    parser.add_argument("--gpu", type=str, default="0", help="gpu card ID")
    args = parser.parse_args()

    print(1)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cudnn.benchmark = True

    train_data, test_data, user_num, item_num, train_mat, poi_data = load_all()
    distances = calculate_distances(poi_data)
    norm_distances = normalize_distances(distances)

    train_dataset = BPRData(train_data, item_num, train_mat, args.num_ng, True)
    test_dataset = BPRData(test_data, item_num, train_mat, 0, False)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = data.DataLoader(test_dataset, batch_size=args.test_num_ng+1, shuffle=False, num_workers=0)

    model = BPR(user_num, item_num, args.factor_num)
    model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.lamda)

    count, best_hr = 0, 0
    for epoch in range(args.epochs):
        model.train()
        start_time = time.time()
        train_loader.dataset.ng_sample()

        for user, item_i, item_j in train_loader:
            user = user.cuda()
            item_i = item_i.cuda()
            item_j = item_j.cuda()
            distance_ij = torch.tensor([norm_distances[i, j] for i, j in zip(item_i.cpu().numpy(), item_j.cpu().numpy())]).cuda()

            model.zero_grad()
            prediction_i, prediction_j, distance_ij = model(user, item_i, item_j, distance_ij)
            loss = - (distance_ij * (prediction_i - prediction_j)).sigmoid().log().sum()
            loss.backward()
            optimizer.step()
            count += 1

        model.eval()
        HR, NDCG = metrics(model, test_loader, args.top_k)

        elapsed_time = time.time() - start_time
        print("The time elapse of epoch {:03d}".format(epoch) + " is: " + 
                time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
        print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))

    #     if HR > best_hr:
    #         best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
    #         if args.out:
    #             if not os.path.exists(path + model_path):
    #                 os.mkdir(path + model_path)
    #             torch.save(model, '{}BPR.pt'.format(path + model_path))

    # print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(best_epoch, best_hr, best_ndcg))
