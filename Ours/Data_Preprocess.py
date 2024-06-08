import json
import csv
import pandas as pd
import numpy as np
from MyBERT import get_Embedding_Using_BERT
from tqdm import tqdm  # tqdm 라이브러리 임포트
import random
from multiprocessing import Pool
import _multiprocessing

def get_business_info(path):
    # JSON 파일 경로
    json_file_path = path + 'business.json'

    # CSV 파일 경로
    csv_file_path = path + 'business_info.csv'

    # CSV 파일을 쓰기 모드로 열기
    with open(csv_file_path, 'w', newline='') as csv_file:
        # CSV 라이터 생성
        csv_writer = csv.writer(csv_file)

        # CSV 파일 헤더 작성
        csv_writer.writerow(['business_id','latitude ','longitute', 'city', 'idx'])

        # business_id 를 int형으로 변환
        idx = 0

        # JSON 파일을 한 줄씩 읽어서 처리
        with open(json_file_path, 'r', encoding='utf-8') as json_file:
            for line in json_file:
                data = json.loads(line)  # JSON 데이터 파싱

                # user_id와 business_id 추출
                business_id = data['business_id']
                latitude = data['latitude']
                longitude = data['longitude']
                city = data['city']

                if city.lower() in 'philadelphia':
                    # CSV 파일에 데이터 작성
                    csv_writer.writerow([business_id, latitude, longitude, city, idx])
                    idx += 1

        print(f"{idx}개의 business_id 생성 완료.")
def get_raw_review(path):
    # JSON 파일 경로
    json_file_path = path + 'review.json'
    # CSV 파일 경로
    csv_file_path = path + 'reviews.csv'
    
    business_info_file = path + 'business_info.csv' # 필라델피아 가게 정보 business_id, latitude, longitude, city, idx

    business_location = {}
    with open(business_info_file, 'r', newline='') as business_file:
        csv_reader = csv.reader(business_file)
        next(csv_reader)  # 헤더 행 건너뛰기
        for row in csv_reader:
            business_id, latitude, longitude, city, idx = row[0], row[1], row[2], row[3].lower(), row[4] #city : 소문자로 받음
            business_location[business_id] = idx
    
    
    # CSV 파일을 쓰기 모드로 열기
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
        # CSV 라이터 생성
        csv_writer = csv.writer(csv_file)
        # CSV 파일 헤더 작성
        csv_writer.writerow(['user_id', 'business_id', 'stars', 'text'])

        # JSON 파일을 한 줄씩 읽어서 처리
        count = 0
        with open(json_file_path, 'r', encoding='utf-8') as json_file:
            for line in json_file:
                data = json.loads(line)  # JSON 데이터 파싱
                # user_id와 business_id 추출
                user_id = data['user_id']
                business_id_str = data['business_id']
                stars = data['stars']
                text = data['text']
                
                if business_id_str in business_location: # 필라델피아 가게에 작성된 리뷰만 검사
                    count += 1
                    idx = business_location[business_id_str]
                    csv_writer.writerow([user_id, idx, stars, text])

    print(f"{count}개의 reviews data 생성 완료.")
def user_id_toInteger(path):
    # CSV 파일 경로
    input_file = path + 'reviews.csv'
    output_file = path + 'reviews.txt'

    # CSV 파일을 읽고 데이터를 리스트로 저장
    data = []
    with open(input_file, 'r', newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader)  # 헤더 행
        #data.append(header)
        for row in csv_reader:
            data.append(row)

    # user_id를 기준으로 데이터를 정렬
    data.sort(key=lambda x: x[0])  # 여기서 0은 user_id 열을 가리킵니다. 0부터 시작하면 첫 번째 열입니다.

    # user_id를 정수형으로 변환
    idx = 0
    before_user_id = data[0][0]
    for i in data:
        if i[0] == before_user_id:
            i[0] = idx
        else:
            idx += 1
            before_user_id = i[0]
            i[0] = idx
            
    # 정렬된 데이터를 새로운 파일에 저장
    with open(output_file, 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(data)

    print("CSV 파일이 user_id를 기준으로 정렬되었고, 새로운 파일에 저장되었습니다.")

def get_data(path = "dataset/"):
    # 위도 경도 load
    business_info_file = path + 'business_info.csv' # 필라델피아 가게 정보 business_id, latitude, longitude, city, idx
    
    business_location = []
    with open(business_info_file, 'r', newline='') as business_file:
        csv_reader = csv.reader(business_file)
        next(csv_reader)  # 헤더 행 건너뛰기
        for row in csv_reader:
            _, latitude, longitude, _, _ = row[0], row[1], row[2], row[3].lower(), row[4] #city : 소문자로 받음

            business_location.append([latitude, longitude])
    business_location = np.array(business_location, dtype=float)

    # 리뷰 load
    input_file = path + "reviews.txt"
    data = []
    with open(input_file, 'r', newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            data.append(row)

    if not data:  # 데이터가 비어있는 경우 처리
        return

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
        if user_id == before_user_id:
            tmp_business_id.append(int(business_id))
            tmp_ratings.append(float(rating))
            tmp_reviews.append(review)
            tmp_review_emb_list.append(idx)
        else:
            if len(tmp_business_id) >= 10:  # 방문 횟수가 10회가 넘는 유저만 append
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
    if len(tmp_business_id) >= 10:
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
    before_poi_id = poi_visited_list[0][1]  # 첫 번째 POI ID로 초기화

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

    item_history_list.append(tmp_business_id)
    item_ratings_list.append(tmp_ratings)
    item_reviews_list.append(tmp_reviews)
    item_review_emb_list.append(tmp_review_emb)

    print(len(item_history_list), len(item_ratings_list), len(item_reviews_list), len(item_review_emb_list))

    # 임베딩 load
    embedding_file = path + 'embeddings.npy'
    embeddings = np.load(embedding_file, mmap_mode='r')

    user_review_embs = []
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

    return user_history_list, user_ratings_list, user_reviews_list, user_review_embs, item_history_list, item_ratings_list, item_reviews_list, item_review_embs, business_location

def get_text_embeddings(path):
    embedding_file = path + 'reviews_embedding.npy'
    
    history, _ , review, item_list, _, item_review = get_data(path)
    
    batch = len(history)//5
    
    history_list = history[:batch]
    reviews_list = review[:batch]
    
    user_embeddings = []
    for user, hist in enumerate(tqdm(history_list)):
        indicies = []
        embeddings = []
        
        while len(indicies) < 10:
            num = random.randint(0,len(hist) - 1)
            if num not in indicies:
                indicies.append(num)
                
        for i in indicies:
            embeddings.append(get_Embedding_Using_BERT(reviews_list[user][i]))
        user_embeddings.append(embeddings)

    item_embeddings = []
    for poi, item in enumerate(tqdm(item_list)):
        indicies = []
        embeddings = []
        
        while len(indicies) < 5:
            num = random.randint(0,len(item) - 1)
            if num not in indicies:
                indicies.append(num)
                
        for i in indicies:
            embeddings.append(get_Embedding_Using_BERT(item_review[poi][i]))
        item_embeddings.append(embeddings)
        

    user_embeddings = np.array(user_embeddings)
    np.save(embedding_file, user_embeddings)


if __name__ == "__main__":
    path = 'dataset/'
    
    #get_business_info(path)
    #get_raw_review(path)
    #user_id_toInteger(path)
    
    #get_text_embeddings(path)
    _,_,_,_,_,_ = get_data(path)
    
    #print(len(data))
    
    
