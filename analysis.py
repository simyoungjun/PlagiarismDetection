#%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from utils import *
import json



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
import re  
def extract_chars(input_string):
    # Define a regular expression pattern to match "NRF," "VC," or "DS"
    pattern = r'(NRF|VC|DS)'

    # Use re.split() to split the string based on the pattern
    result = re.split(pattern, input_string)

    # Filter out empty strings from the result
    result = [item for item in result if item]

    return result
#%
from collections import Counter

scores_dict = {}
for np_path, np_array in zip(['y_cosine', 'y_euclidean', 'y_corr', 'y_test'], [y_cosine, y_euclidean, y_corr, y_test]):
    scores_dict[np_path] = np.load('./test_scores/'+np_path+'.npy')

y_cosine, y_euclidean, y_corr, y_test = scores_dict['y_cosine'], scores_dict['y_euclidean'], scores_dict['y_corr'], scores_dict['y_test'],

with open('./data/test.json', 'rt', encoding='UTF8') as json_file:
        dataset = json.load(json_file)

y_test_ = [dataset[i]['is_plagiarism'] for i in range(len(dataset))]


for y_prob, metric, optimal_threshold in zip([y_cosine, y_corr, y_euclidean],['Cosine', 'Corr', 'Euclidean'], optimal_threshold_list):
    if metric == 'Euclidean':
        y_test = [1 if item == 0 else 0 for item in y_test]
    # 정확도 계산
    y_pred = (y_prob >= optimal_threshold).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"최적 임계값을 기반으로 한 Test set 정확도: {accuracy:.6f}")
    
    #틀린 샘플의 idx
    wrong_indices = np.where(y_pred != y_test)[0]
    
    wrong_pair_list = [dataset[i]['id'] for i in wrong_indices]
    wrong_pair_list_label = [dataset[i]['is_plagiarism'] for i in wrong_indices]
    
    # Your list
    keyword_list = []
    count_nrf = 0
    count_vc = 0
    count_ds = 0
    for wrong_pair, wrong_pair_list_label in zip(wrong_pair_list, wrong_pair_list_label):
        result = extract_chars(wrong_pair)

        for item in result:
            if 'NRF' in item:
                count_nrf += 1
                # print(wrong_pair)
            if 'VC' in item:
                count_vc += 1
                # print(wrong_pair)
            if 'DS' in item:
                count_ds += 1
                print(wrong_pair)
                
                
    print(count_nrf, count_vc, count_ds)
    wrong = [count_nrf, 0, 0, count_vc, 0, count_ds]
    wrong = [value / 2 for value in wrong]

    
    labels = ['NRF-NRF', 'NRF-VC', 'NRF-DS', 'VC-VC', 'VC-DS', 'DS-DS']
    # Create a bar chart
    plt.bar(labels, wrong)

    # Add labels and a title
    plt.xlabel('Substrings')
    plt.ylabel('Count')
    plt.title('Counts of Wrong in '+f"{metric}")
    plt.show()
    
    print(len(wrong_pair_list))
    