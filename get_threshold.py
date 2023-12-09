#%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from utils import *
import json
from tqdm import tqdm
#%

# 모델 학습
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")

#%
with open('./data/test.json', 'rt', encoding='UTF8') as json_file:
        dataset = json.load(json_file)

# 클래스 1에 대한 확률을 예측

y_cosine = []
y_euclidean = []
y_corr = []
y_test = []
for i, data in enumerate(tqdm(dataset)):
    emb1 = get_embedding(data['content1'])
    emb2 = get_embedding(data['content2'])
    y_cosine.append(cosine_similarity(emb1, emb2))
    y_euclidean.append(euclidean_distance(emb1, emb2))
    y_corr.append(correlation_coefficient(emb1, emb2))
    y_test.append(data['is_plagiarism'])
    
    # if i % 500 == 0:
    #     for np_path, np_array in zip(['y_cosine', 'y_euclidean', 'y_corr', 'y_test'], [y_cosine, y_euclidean, y_corr, y_test]):
    #         np.save('./test_scores/'+np_path, np_array)


test_prob_emb = {'y_cosine': y_cosine, 'y_corr': y_corr, 'y_euclidean': y_euclidean, 'y_test': y_test}

# for np_path, np_array in zip(['y_cosine', 'y_euclidean', 'y_corr', 'y_test'], [y_cosine, y_euclidean, y_corr, y_test]):
#     np.save('./test_scores/'+np_path, np_array)
    
#%
with open('./data/train.json', 'rt', encoding='UTF8') as json_file:
        dataset = json.load(json_file)

# 클래스 1에 대한 확률을 예측

y_cosine = []
y_euclidean = []
y_corr = []
y_test = []
for i, data in enumerate(tqdm(dataset)):
    emb1 = get_embedding(data['content1'])
    emb2 = get_embedding(data['content2'])
    y_cosine.append(cosine_similarity(emb1, emb2))
    y_euclidean.append(euclidean_distance(emb1, emb2))
    y_corr.append(correlation_coefficient(emb1, emb2))
    y_test.append(data['is_plagiarism'])
    
#     if i % 500 == 0:
#         for np_path, np_array in zip(['y_cosine', 'y_euclidean', 'y_corr', 'y_test'], [y_cosine, y_euclidean, y_corr, y_test]):
#             np.save('./train_scores/'+np_path, np_array)


# for np_path, np_array in zip(['y_cosine', 'y_euclidean', 'y_corr', 'y_test'], [y_cosine, y_euclidean, y_corr, y_test]):
#     np.save('./train_scores/'+np_path, np_array)



#%
scores_dict = {}
for np_path in ['y_cosine', 'y_euclidean', 'y_corr', 'y_test']:
    scores_dict[np_path] = np.load('./train_scores/'+np_path+'.npy')

y_cosine, y_euclidean, y_corr, y_test = scores_dict['y_cosine'], scores_dict['y_euclidean'], scores_dict['y_corr'], scores_dict['y_test'],

# AUC 계산
optimal_threshold_list = []
for y_prob, metric in zip([y_cosine, y_corr, y_euclidean],['Cosine', 'Corr', 'Euclidean']):
    
    if metric == 'Euclidean':
        y_test = [1 if item == 0 else 0 for item in y_test]
    
    roc_auc = roc_auc_score(y_test, y_prob)

    # ROC 곡선을 계산
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    threshold_idx = np.argmax(tpr - fpr)
    optimal_threshold_list.append(optimal_threshold)
    # 정확도 계산
    y_pred = (y_prob >= optimal_threshold).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    # ROC 곡선 그리기
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # 임계값에 해당하는 점 찍기
    optimal_fpr = fpr[threshold_idx]
    optimal_tpr = tpr[threshold_idx]
    plt.plot(optimal_fpr, optimal_tpr, marker='o', color='red', label=f'Optimal Threshold = {optimal_threshold:.2f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(metric+' ROC Curve (ACC: '+f"{accuracy: .3f}"+',  AUC: '+f"{accuracy: .3f}"+')')
    plt.legend(loc='lower right')

    # AUC 값 출력

    # 최적의 임계값 찾기
    # optimal_threshold = thresholds[np.argmax(tpr - fpr)]


    print(metric)
    print("AUC:", roc_auc)
    print(f"최적의 임계값: {optimal_threshold:.6f}")
    print(f"최적 임계값을 기반으로 한 Train set 정확도: {accuracy:.6f}")
    plt.show()
    
#% 
scores_dict = {}
for np_path, np_array in zip(['y_cosine', 'y_euclidean', 'y_corr', 'y_test'], [y_cosine, y_euclidean, y_corr, y_test]):
    scores_dict[np_path] = np.load('./test_scores/'+np_path+'.npy')

y_cosine, y_euclidean, y_corr, y_test = scores_dict['y_cosine'], scores_dict['y_euclidean'], scores_dict['y_corr'], scores_dict['y_test'],

with open('./data/test.json', 'rt', encoding='UTF8') as json_file:
        dataset = json.load(json_file)

y_test_ = [y for y in dataset[i]['is_plagiarism']]

for y_prob, metric, optimal_threshold in zip([y_cosine, y_corr, y_euclidean],['Cosine', 'Corr', 'Euclidean'],optimal_threshold_list):
    if metric == 'Euclidean':
        y_test = [1 if item == 0 else 0 for item in y_test]
    # 정확도 계산
    y_pred = (y_prob >= optimal_threshold).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"최적 임계값을 기반으로 한 Test set 정확도: {accuracy:.6f}")
    
    #틀린 샘플의 idx
    wrong_indices = np.where(y_pred != y_test)[0]
    