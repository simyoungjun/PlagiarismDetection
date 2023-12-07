#%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from utils import *
import json

with open('./data/train.json', 'r') as json_file:
        dataset = json.load(json_file)

# 모델 학습
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")

# 클래스 1에 대한 확률을 예측

y_prob = []
y_test = []
for data in dataset:
    y_prob.append(get_similarity(data['content1'], data['content2'], metric = 'cosine'))
    y_test.append(data['is_plagiarism'])
# y_prob = model.predict_proba(X_test)[:, 1]

# AUC 계산
roc_auc = roc_auc_score(y_test, y_prob)

# ROC 곡선을 계산
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# ROC 곡선 그리기
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')

# AUC 값 출력
print("AUC:", roc_auc)

plt.show()