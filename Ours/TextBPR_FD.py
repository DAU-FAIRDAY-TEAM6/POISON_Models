from __init__ import *
class TextBPR(nn.Module):
    '''
    MF 모델에 대한 BPR 학습 
    '''
    def __init__(self, dataset, factors, text_factors, learning_rate, reg, init_mean, init_stdev, alpha1, alpha2):
        '''
        생성자
        Args:
            dataset: 데이터셋 객체로, 학습 및 테스트 데이터를 포함합니다.
            factors (int): 잠재 요인의 수.
            learning_rate (float): 최적화에 사용되는 학습률.
            reg (float): 정규화 강도.
            init_mean (float): 초기화에 사용되는 정규 분포의 평균.
            init_stdev (float): 초기화에 사용되는 정규 분포의 표준 편차.
        '''
        super(TextBPR, self).__init__()
        self.dataset = dataset
        self.train_data = dataset.train
        self.test_data = dataset.test
        self.test_for_eval = dataset.test_for_eval
        self.num_user = dataset.num_user
        self.num_item = dataset.num_item
        self.neg = dataset.neg
        self.factors = factors
        self.factors_Text = text_factors
        self.learning_rate = learning_rate
        self.reg = reg
        self.init_mean = init_mean
        self.init_stdev = init_stdev
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        
        self.norm_distances = dataset.norm_distances
        self.norm_frequencys = dataset.norm_frequencys

        self.user_review_embeds = dataset.user_review_embeds
        self.poi_review_embeds = dataset.poi_review_embeds
        
        
        # 사용자와 아이템의 잠재 요인을 초기화합니다.
        self.embed_user = torch.normal(mean=self.init_mean * torch.ones(self.num_user, self.factors), std=self.init_stdev).to(DEVICE).requires_grad_()
        self.embed_item = torch.normal(mean=self.init_mean * torch.ones(self.num_item, self.factors), std=self.init_stdev).to(DEVICE).requires_grad_()
        
        self.beta_items = torch.normal(mean=self.init_mean * torch.ones(self.num_item, 1), std=self.init_stdev).to(DEVICE).requires_grad_()
        self.text_bias = torch.normal(mean=self.init_mean * torch.ones(768, 1), std=self.init_stdev).to(DEVICE).requires_grad_()
        
        # Adam optimizer를 초기화합니다.
        self.mf_optim = optim.Adam([self.embed_user, self.embed_item, self.beta_items, self.text_bias], lr=self.learning_rate, weight_decay=1e-5)

    def forward(self, u, i, j):
        '''
        MF-BPR 모델의 forward pass입니다.
        Args:
            u: 사용자 ID.
            i: 긍정적인 아이템 ID.
            j: 부정적인 아이템 ID.
        Returns:
            y_ui: 사용자와 긍정적인 아이템 간의 예측 점수.
            y_uj: 사용자와 부정적인 아이템 간의 예측 점수.
            loss: BPR 손실.
        '''
        # 사용자와 긍정적인 아이템 간의 예측 점수 계산
        
        user_latent_factor = self.embed_user[u]
        user_text_factors = self.user_review_embeds[u] / math.sqrt(786)
        
        
        i_bias = self.beta_items[i] # batch * 1
        j_bias = self.beta_items[j] # batch * 1
        
        i_text_factors = self.poi_review_embeds[i] # batch * 768
        j_text_factors = self.poi_review_embeds[j] # batch * 768
        
        i_latent_factors = self.embed_item[i]
        j_latent_factors = self.embed_item[j]
        
        diff_latent_factors = i_latent_factors - j_latent_factors # batch * latent
        diff_text_factors = (i_text_factors - j_text_factors) / math.sqrt(768) # batch * 768
    
        if diff_text_factors.shape[0] == 768: # [768], eval set이라면
            user_latent_factor = user_latent_factor.unsqueeze(0)
            user_text_factors = user_text_factors.unsqueeze(0)
            diff_text_factors = diff_text_factors.unsqueeze(0) # [1, text_emb]
            diff_latent_factors = diff_latent_factors.unsqueeze(0) # [ 1, latent_emb]
        
        latent_factor = (user_latent_factor * diff_latent_factors).sum(dim=-1).unsqueeze(-1)
        text_factor = (user_text_factors * diff_text_factors).sum(dim=-1).unsqueeze(-1)
        
        u_i_score = self.alpha1 * latent_factor + (1-self.alpha1) * text_factor
        text_bias = diff_text_factors.mm(self.text_bias)
        
        distance_ij = torch.tensor(self.norm_distances[i, j]).to(DEVICE)
        frequency_ij = torch.tensor([self.norm_frequencys.get((i[idx].item(), j[idx].item()), 0.5) for idx in range(len(i))]).to(DEVICE)
        wij = (self.alpha2 * distance_ij) + ((1-self.alpha2) * frequency_ij)
        
        x_uij = i_bias - j_bias + (wij * u_i_score) + text_bias
        
        # 정규화 항 계산
        # BPR 손실 계산
        loss = -torch.sum(torch.log(torch.sigmoid(x_uij.unsqueeze(0))))
        return loss

    def build_model(self, epoch=30, batch_size=32, topK = 10):
        '''
        MF-BPR 모델을 구축하고 학습합니다.
        Args:
            epoch (int): 학습의 최대 반복 횟수.
            num_thread (int): 병렬 실행을 위한 스레드 수.
            batch_size (int): 학습용 배치 크기.
        '''
        data_loader = DataLoader(self.dataset, batch_size=batch_size)

        print("Training MF-BPR with: learning_rate=%.4f, regularization=%.7f, factors=%d, #epoch=%d, batch_size=%d, alpha1=%.2f, alpha2=%.2f."
               % (self.learning_rate, self.reg, self.factors, epoch, batch_size, alpha1, alpha2))
        t1 = time.time()
        
        max_hit, max_precision, max_recall, max_recall_epoch, max_precision_epoch, max_hit_epoch = 0,0,0,0,0,0
        for epoc in range(epoch):
            self.train()
            iter_loss = 0
            count = 0
            for s, (users, items_pos, items_negs) in enumerate(data_loader):
                count += 1
                # 기울기 초기화
                self.mf_optim.zero_grad()
                # Forward pass를 통해 예측과 손실 계산
                for items_neg in items_negs:
                    loss = self.forward(users, items_pos, items_neg)
                    iter_loss += loss 
                    # Backward pass 및 파라미터 업데이트
                    loss.backward()
                self.mf_optim.step()
            t2 = time.time()
            
            # 성능 측정 함수를 통해 HitRatio 및 NDCG를 계산
            if epoc % 1 == 0:
                self.eval()
                hits, recall, precision = self.evaluate_model(self.test_data, topK)
                eval_loss = 0
                # for idx, (u, i, j) in enumerate(self.test_for_eval):
                #     loss = self.forward(u, i, j)
                #     eval_loss += loss
                total_samples = len(self.test_for_eval)
                eval_loss = eval_loss / total_samples if total_samples > 0 else 0
                iter_loss = iter_loss / count / batch_size
                print(f"epoch={epoc}, train_loss = {iter_loss}, test_loss = {eval_loss}[{t2-t1}s] HitRatio@{topK} = {hits}, RECAll@{topK} = {recall}, PRECISION@{topK} = {precision} [{time.time()-t2}s]")
                t1 = time.time()
                if precision > max_precision:
                    max_precision = precision
                    max_precision_epoch = epoc
                if recall > max_recall:
                    max_recall = recall
                    max_recall_epoch = epoc
                if hits > max_hit:
                    max_hit = hits
                    max_hit_epoch = epoc
                t1 = time.time()
                
        save_perform(reg, batch_size, latent_factors, text_factors, epoc, learning_rate, max_hit, max_hit_epoch, max_recall, max_recall_epoch, max_precision, max_precision_epoch, alpha1, alpha2)
    
    
    def evaluate_model(self, test, K):
        """
        Top-K 추천의 성능(Hit_Ratio, NDCG)을 평가합니다.   
        반환값: 각 테스트 상호작용의 점수.
        """
        user_latent_factor = self.embed_user # batch * latent
        item_latent_factors = self.embed_item # batch * latent
            
        user_text_factors = self.user_review_embeds / math.sqrt(768) # batch * latent
        item_text_factors = self.poi_review_embeds / math.sqrt(768)# batch * 768
            

        latent_score_matrix = torch.mm(user_latent_factor, item_latent_factors.t())
        text_score_matrix = torch.mm(user_text_factors, item_text_factors.t())
        
        score_matrix = self.alpha1 * latent_score_matrix + (1-self.alpha1) * text_score_matrix
        
        item_bias = self.beta_items.squeeze()
        item_bias = item_bias.view(1, -1)
        score_matrix = score_matrix + item_bias
        
        
        top_scores, top_indicies = torch.topk(score_matrix, K, dim=1)
        
        hits = 0
        sum_recall = 0
        sum_precision = 0 
        for u,hist in enumerate(test):
            set_topk = set(i.item() for i in (top_indicies[u]))
            set_hist = set(hist)
            
            if set_hist & set_topk:
                hits += 1
            sum_precision += len(set_hist & set_topk) / len(set_topk)
            sum_recall += len(set_hist & set_topk) / len(set_hist)
            
        return hits / len(test), sum_recall / len(test), sum_precision / len(test)

class Yelp(Dataset):
    def __init__(self):
        """
        Yelp 데이터셋을 로드하고 학습 데이터와 테스트 데이터를 생성합니다.

        Args:
            dir (str): 데이터 파일이 있는 디렉토리 경로.
            splitter (str): 파일에서 열을 구분하는 구분자.
            K (int): K 값, 즉 각 사용자마다 테스트에 사용되는 상호작용의 수.
        """
        path = 'dataset/'
        user_history_list, _, _, user_review_embeds ,_,_,_,poi_review_embeds, business_location, user_frequency_list = dp.get_data(path)
        self.norm_distances = self.normalize_distances(self.calculate_distances(business_location))
        self.norm_frequencys = self.normalize_frequency(self.calculate_frequency(user_frequency_list))

        self.train = []
        self.test = []
        self.poi_review_embeds = torch.tensor(poi_review_embeds).to(DEVICE)
        self.user_review_embeds = torch.tensor(user_review_embeds).to(DEVICE)
        self.num_user = len(user_history_list)
        self.num_item = len(poi_review_embeds) # 14585

        items = [i for i in range(self.num_item)]
        self.neg = dict()

        random.seed(30)
        for u, hist in enumerate(user_history_list):
            random.shuffle(hist)
            self.train.append(hist[:int(len(hist) * 0.7)])
            self.test.append(hist[int(len(hist) * 0.7) :])

            u_negs = set(items) - set(hist)
            self.neg[u] = list(u_negs) # ng dataset 생성

        self.test_for_eval = []
        for u,hist in enumerate(self.test):
            for i in hist:
                self.test_for_eval.append([u,i])

        self.index_map = []
        for u, user_items in enumerate(self.train):
            for i in user_items:
                self.index_map.append((u, i))

    def __len__(self):
        """
        데이터셋의 사용자 수를 반환합니다.
        """
        #return self.num_user
        return len(self.index_map)

    def __getitem__(self, idx):
        """
        데이터셋에서 하나의 샘플을 가져옵니다.

        Args:
            idx (int): 데이터셋 내의 인덱스.

        Returns:
            u: 사용자 ID.
            i: 긍정적인 아이템 ID.
            j: 부정적인 아이템 ID.
        """
        # u = idx
        # # 사용자별로 하나의 긍정적인 상호작용 선택
        # i = self.train[u][np.random.randint(0, len(self.train[u]))]
        # # 부정적인 상호작용 무작위 선택
        # j = self.neg[u][np.random.randint(0, len(self.neg[u]))]
        # #j = random.sample(self.neg[u], 4)
        # return (u, i, j)

        u, i = self.index_map[idx]
        # 부정적인 아이템 무작위 선택
        j = random.sample(self.neg[u], 4)
        #j = self.neg[u][np.random.randint(0, len(self.neg[u]))]
        return (u, i, j)

    def haversine(self, lat1, lon1, lat2, lon2):
        R = 6371
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon1 - lon2)  # Note the change here
        a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        d = R * c
        return d

    def calculate_distances(self, poi_data):
        t = time.time()
        distances = squareform(pdist(poi_data, lambda u, v: self.haversine(u[0], u[1], v[0], v[1])))
        print(f"calculate_distance time : {int(time.time()-t)}")
        return distances

    def normalize_distances(self, distances):
        min_d = np.min(distances)
        max_d = np.max(distances)
        norm_distances = 0.5 * (distances - min_d) / (max_d - min_d) + 0.5
        return norm_distances
    
    def calculate_frequency(self, frequency_list):
        frequency_diff = {}
        for u, freq_dict in enumerate(frequency_list):
            pois = list(freq_dict.keys())
            for i in pois:
                for j in pois:
                    if i != j:
                        fui = freq_dict.get(i, 0)  # POI i의 빈도수
                        fuj = freq_dict.get(j, 0)  # POI j의 빈도수
                        fuij = fui - fuj  # 빈도수 차이
                        business_pair = (i, j)
                        frequency_diff[business_pair] = fuij
        return frequency_diff

    def normalize_frequency(self, frequency_diff):
        min_freq_diff = min(frequency_diff.values())  # 최소 빈도수 차이
        max_freq_diff = max(frequency_diff.values())  # 최대 빈도수 차이
        norm_frequencys = {}
        for key, fuij in frequency_diff.items():
            norm_fuij = 0.5 * ((fuij - min_freq_diff) / (max_freq_diff - min_freq_diff)) + 0.5
            norm_frequencys[key] = norm_fuij
        return norm_frequencys

def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0
def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0


if __name__ == '__main__':
    dataset = "POISON_Models\BPR_Model\Ours\data/"  # 데이터셋 디렉토리 또는 파일
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device = {DEVICE}")
    yelp = Yelp()
    latent_factors = args.latent_factors
    text_factors = args.text_factors
    learning_rate = args.lr  # 학습률
    reg = args.reg  # 정규화 계수
    epoch = args.epoch
    batch_size = args.batch_size  # 미니배치 크기
    alpha1 = args.alpha1
    alpha2 = args.alpha2
    init_mean = 0  # 초기 가중치 평균
    init_stdev = 0.001  # 초기 가중치 표준편차
    
    K = 10
    print(latent_factors, text_factors, learning_rate, reg, epoch, batch_size)
    
    print("#factors: %d, lr: %f, reg: %f, batch_size: %d" % (latent_factors, learning_rate, reg, batch_size))
    
    # MF-BPR 모델 생성 및 학습
    text_bpr = TextBPR(yelp, latent_factors, text_factors, learning_rate, reg, init_mean, init_stdev, alpha1, alpha2)
    text_bpr.build_model(epoch, batch_size=batch_size, topK = K)

    # 학습된 가중치 저장
    #np.save("out/u"+str(learning_rate)+".npy", bpr.U.detach().numpy())
    #np.save("out/v"+str(learning_rate)+".npy", bpr.V.detach().numpy())