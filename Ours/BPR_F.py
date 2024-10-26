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
    path = './Ours/dataset/'
    
    train_data = pd.read_csv(
        path + 'train_freq.csv', 
        sep='\t', header=None, names=['user', 'item', 'freq'], 
        usecols=[0, 1, 2], dtype={0: np.int32, 1: np.int32, 2: np.int32})

    user_num = train_data['user'].max() + 1
    item_num = train_data['item'].max() + 1

    train_data = train_data.values.tolist()

    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for x in train_data:
        train_mat[x[0], x[1]] = x[2]  # 빈도수로 설정

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

    return train_data, test_data, user_num, item_num, train_mat


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
            u, i, _ = x[0], x[1], x[2]
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
        freq_i = self.train_mat[user, item_i]
        item_j = features[idx][2] if self.is_training else features[idx][1]
        freq_j = self.train_mat[user, item_j] if self.is_training else 0
        return user, item_i, item_j, freq_i, freq_j


class BPR(nn.Module):
    def __init__(self, user_num, item_num, factor_num):
        super(BPR, self).__init__()
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

    def forward(self, user, item_i, item_j):
        user = self.embed_user(user)
        item_i = self.embed_item(item_i)
        item_j = self.embed_item(item_j)
        prediction_i = (user * item_i).sum(dim=-1)
        prediction_j = (user * item_j).sum(dim=-1)
        return prediction_i, prediction_j


def hit(gt_item, pred_items):
    if gt_item in pred_items:
        return 1
    return 0


def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index + 2))
    return 0


def metrics(model, test_loader, top_k):
    HR, NDCG = [], []

    for user, item_i, item_j, _, _ in test_loader:
        user = user.cuda()
        item_i = item_i.cuda()
        item_j = item_j.cuda()

        prediction_i, prediction_j = model(user, item_i, item_j)
        _, indices = torch.topk(prediction_i, top_k)
        recommends = torch.take(item_i, indices).cpu().numpy().tolist()

        gt_item = item_i[0].item()
        HR.append(hit(gt_item, recommends))
        NDCG.append(ndcg(gt_item, recommends))

    return np.mean(HR), np.mean(NDCG)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.05, help="learning rate")
    parser.add_argument("--lamda", type=float, default=0.005, help="model regularization rate")
    parser.add_argument("--batch_size", type=int, default=4096, help="batch size for training")
    parser.add_argument("--epochs", type=int, default=300, help="training epoches")
    parser.add_argument("--top_k", type=int, default=10, help="compute metrics@top_k")
    parser.add_argument("--factor_num", type=int, default=32, help="predictive factors numbers in the model")
    parser.add_argument("--num_ng", type=int, default=4, help="sample negative items for training")
    parser.add_argument("--test_num_ng", type=int, default=99, help="sample part of negative items for testing")
    parser.add_argument("--out", default=True, help="save model or not")
    parser.add_argument("--gpu", type=str, default="0", help="gpu card ID")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cudnn.benchmark = True

    train_data, test_data, user_num, item_num, train_mat = load_all()

    train_dataset = BPRData(train_data, item_num, train_mat, args.num_ng, True)
    test_dataset = BPRData(test_data, item_num, train_mat, 0, False)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = data.DataLoader(test_dataset, batch_size=args.test_num_ng + 1, shuffle=False, num_workers=0)

    model = BPR(user_num, item_num, args.factor_num)
    model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.lamda)

    count, best_hr = 0, 0
    with open('training_log_F.txt', 'w') as f:
        for epoch in range(args.epochs):
            model.train()
            start_time = time.time()
            train_loader.dataset.ng_sample()

            for user, item_i, item_j, freq_i, freq_j in train_loader:
                user = user.cuda()
                item_i = item_i.cuda()
                item_j = item_j.cuda()
                freq_i = freq_i.cuda()
                freq_j = freq_j.cuda()

                model.zero_grad()
                prediction_i, prediction_j = model(user, item_i, item_j)
                freq_diff = (freq_i - freq_j).float()
                min_f, max_f = freq_diff.min().item(), freq_diff.max().item()
                w_uij = 0.5 * ((freq_diff - min_f) / (max_f - min_f)) + 0.5
                loss = - (w_uij * (prediction_i - prediction_j).sigmoid().log()).sum()
                loss.backward()
                optimizer.step()
                count += 1

            model.eval()
            HR, NDCG = metrics(model, test_loader, args.top_k)

            elapsed_time = time.time() - start_time
            epoch_time_str = f"The time elapse of epoch {epoch:03d} is: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            metrics_str = f"HR: {np.mean(HR):.3f}\tNDCG: {np.mean(NDCG):.3f}"

            print(epoch_time_str)
            print(metrics_str)
            f.write(epoch_time_str + "\n")
            f.write(metrics_str + "\n")
