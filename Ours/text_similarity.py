from __init__ import *

def get_review_embs(path):
    input_file = path + "reviews.txt"
    data = []
    with open(input_file, 'r', newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            data.append(row)
    
    reviews = []
    
    user_history_list = []
    user_reviews_list = []
    user_ratings_list = []
    user_review_emb_list = []
    
    tmp_reviews = []
    tmp_ratings = []
    tmp_business_id = []
    tmp_review_emb_list = []
    
    before_user_id = data[0][0]  # 첫 번째 사용자 ID로 초기화
    
    
    for idx, i in enumerate(data):
        user_id, business_id, rating, review = i
        reviews.append(review)
        if user_id == before_user_id:
            tmp_business_id.append(int(business_id))
            tmp_ratings.append(float(rating))
            tmp_reviews.append(review)
            tmp_review_emb_list.append(idx)
        else:
            if len(tmp_business_id) >= 3 and len(tmp_business_id) <10:  # 방문 횟수가 3회 이상, 15회 이하인 유저만 append
                user_history_list.append(tmp_business_id)
                user_ratings_list.append(tmp_ratings)
                user_reviews_list.append(tmp_reviews)
                user_review_emb_list.append(tmp_review_emb_list)
            tmp_business_id = [int(business_id)]
            tmp_ratings = [float(rating)]
            tmp_reviews = [review]
            tmp_review_emb_list = [idx]
            
            before_user_id = user_id  # 현재 사용자 ID로 업데이트

    # 마지막 사용자 처리
    if len(tmp_business_id) >= 3 and len(tmp_business_id) <10:
        user_history_list.append(tmp_business_id)
        user_ratings_list.append(tmp_ratings)
        user_reviews_list.append(tmp_reviews)
        user_review_emb_list.append(tmp_review_emb_list)
    print(len(user_history_list), len(user_ratings_list), len(user_reviews_list), len(user_review_emb_list))
    
    
    # POI가 가진 리뷰 임베딩을 획득하기 위해
    # history_list를 기준으로 POI에 방문한 사람들 list 생성
    poi_visited_list = []
    for user,history in enumerate(user_history_list):
        for idx, poi in enumerate(history):
            poi_visited_list.append([int(user), int(poi), float(user_ratings_list[user][idx]), user_reviews_list[user][idx], user_review_emb_list[user][idx]])
            
    poi_visited_list.sort(key = lambda x:x[1]) # poi 번호 순으로 정렬
    
    
    item_history_list = []
    item_reviews_list = []
    item_ratings_list = []
    item_review_emb_list = []
    
    
    tmp_reviews = []
    tmp_ratings = []
    tmp_user_id = []
    tmp_review_emb = []
    before_poi_id = poi_visited_list[0][1]  # 첫 번째 사용자 ID로 초기화
    
    for idx, i in enumerate(poi_visited_list):
        user_id, business_id, rating, review, review_emb = i[0], i[1], i[2], i[3], i[4]
        if business_id == before_poi_id: # 이전 POI Id와 동일하다면
            tmp_user_id.append(user_id)
            tmp_ratings.append(rating)
            tmp_reviews.append(review)
            tmp_review_emb.append(review_emb)
        else: # 이전 POI ID와 다른 POI라면
            #print(business_id)
            # 이전 POI 정보 안에 있던거 다 추가하고
            item_history_list.append(tmp_user_id)
            item_ratings_list.append(tmp_ratings)
            item_reviews_list.append(tmp_reviews)
            item_review_emb_list.append(tmp_review_emb)
            
            if int(business_id) - int(before_poi_id) > 1:
                for _ in range(int(business_id) - int(before_poi_id) - 1):
                    #print(f"방문 기록이 없는 POI는 PASS")
                    item_history_list.append([])
                    item_ratings_list.append([])
                    item_reviews_list.append([])
                    item_review_emb_list.append([])
                    
            tmp_user_id = [user_id]
            tmp_ratings = [rating]
            tmp_reviews = [review]
            tmp_review_emb = [review_emb]
            
            before_poi_id = business_id  # 현재 사용자 ID로 업데이트
         
    #마지막 사용자 처리
    item_history_list.append(tmp_business_id)
    item_ratings_list.append(tmp_ratings)
    item_reviews_list.append(tmp_reviews)
    item_review_emb_list.append(tmp_review_emb)
    
    
    embedding_file = path + 'embeddings.npy'
    embeddings = np.load(embedding_file, mmap_mode='r')
    
    user_review_embs = [] # 사용자를 대표하는 임베딩
    for poi, embeds in enumerate(user_review_emb_list):
        if len(embeds)>0: # 비어있지 않으면
            new_array = np.array([embeddings[idx] for idx in embeds])
            new_array = np.mean(new_array, axis = 0)
        else:
            new_array = np.zeros(768, dtype=np.float32)
            
        user_review_embs.append(new_array)


    item_review_embs = []
    for poi, embeds in enumerate(item_review_emb_list):
        if len(embeds)>0: # 비어있지 않으면
            new_array = np.array([embeddings[idx] for idx in embeds])
            new_array = np.mean(new_array, axis = 0)
        else:
            new_array = np.zeros(768, dtype=np.float32)
            
        item_review_embs.append(new_array.tolist())


    return user_review_embs, user_reviews_list, item_review_embs, item_reviews_list, embeddings, reviews


def get_top_similar_reviews(user_review_emb, all_reviews_emb, top_n=10):
    similarities = []
    for idx, review_emb in enumerate(all_reviews_emb):
        similarity = 1 - cosine(user_review_emb, review_emb)
        similarities.append((similarity, idx))
    
    # Sort based on similarity (higher is better) and get the top_n
    similarities.sort(reverse=True, key=lambda x: x[0])
    
    top_indices = [idx for _, idx in similarities[:top_n]]
    top_similarities = [sim for sim, _ in similarities[:top_n]]
    
    
    return top_indices, top_similarities

if __name__ == "__main__":
    path = 'dataset/'
    
    # input_file = path + "reviews.txt"
    # data = []
    # count = 0
    # with open(input_file, 'r', newline='', encoding='utf-8') as csv_file:
    #     csv_reader = csv.reader(csv_file)
    #     for row in csv_reader:
    #         data.append(row)   
    #         if len(row[3]) > 1500:
    #             count += 1
    
    # print (count)
    user_review_embs, user_reviews_list, item_review_embs, item_reviews_list, review_embeddings, reviews = get_review_embs(path)
    
    users = [2]#list(range(2))
    results = {}
    
    for user in users:
        user_emb = user_review_embs[user]
        top_indices, top_similarities = get_top_similar_reviews(user_emb, review_embeddings)
        
        user_real_reviews = user_reviews_list[user]
        user_similar_reviews = [reviews[i] for i in top_indices]
        results[user] = {'similarities': top_similarities, 'real_review': user_real_reviews, 'similar_review' : user_similar_reviews}

        
        
    # Example of accessing the results for user 0
    u = 2
    print("User : ", u)
    print("User 0 top 10 similar reviews similarities:", results[u]['similarities'])
    
    print("@" * 20)
    print("User real review:",  "\n @".join(results[u]['real_review']))
    print("@" * 20)
    print("User similar review :",  "\n @".join(results[u]['similar_review']))
    print("@" * 20)
    
    # pois = [1]#list(range(2))
    # results = {}
    
    # for poi in pois:
    #     item_emb = item_review_embs[poi]
    #     top_indices, top_similarities = get_top_similar_reviews(item_emb, review_embeddings)
        
    #     item_real_reviews = item_reviews_list[poi]
    #     item_similar_reviews = [reviews[i] for i in top_indices]
    #     results[poi] = {'similarities': top_similarities, 'real_review': item_real_reviews, 'similar_review' : item_similar_reviews}
        
    # # Example of accessing the results for user 0
    # i = 1
    # print("Item : ", i)
    # print("Item 0 top 10 similar reviews similarities:", results[i]['similarities'])
    
    # print("@" * 20)
    # print("Item real review:",  "\n @".join(results[i]['real_review']))
    # print("@" * 20)
    # print("Item similar review :",  "\n @".join(results[i]['similar_review']))
    # print("@" * 20)