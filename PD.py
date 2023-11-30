import json
from utils import *
from nltk import sent_tokenize



#TODO: change this to read data
with open('temp.json', 'r') as f:
    data = json.load(f)

augmented_data = []

THRESHOLD = 0.5

for i in range(len(data)):
    abstract = data[i]['abstract']
    title = data[i]['title']
    # Split the abstract into sentences as a list
    paragraph = sent_tokenize(abstract)
    data_id = data[i]['id']
    '''
    In the utils.py, we have various functions to augment the data
    synonym_replacement: replace words with their synonyms
    random_insertion: insert random synonyms into the sentence
    random_swap: swap words randomly
    random_deletion: delete words randomly
    paraphrase: paraphrase uses the parrot model to paraphrase the sentence
    '''
    
    # For example, if you want to use synonym_replacement, you can do the following:
    # rate is the rate of replacement. 0.1 means 10% of the words will be replaced in the sentence
    modified_abstracts = synonym_replacement(paragraph, rate = 0.1, num_aug = 1)
    augmented_data.append({'id': data_id, 'title': title, 'abstract': abstract})

    for i, modified_abstract in enumerate(modified_abstracts):
        augmented_data.append({'id': int(str(data_id)+'_'+str(i)), 'title': title, 'abstract': modified_abstract})

#TODO: change this to write to the file name
with open('augmented_data.json', 'w') as f:
    json.dump(augmented_data, f)


'''
데이터 다 수집하고 나면,
위의 코드로 각자 자기 주제에 맞는 데이터를
synonym, insertion, swap, deletion, paraphrase 다섯개의 방법을 이용해서 augmentation (num_aug = 1로 하면 될듯)

다 만들고 나면 pair 별 similarity를 구해서 is_plagiarism 구하기
threshold 바꿔가면서 실험해볼 것

data id 형식을 {몇번째 논문}_{몇번째 augmentation} 으로 지정해놨으므로,
같은 파일 내의 논문{몇번째 논문}이 같음: is_plariarsim = 1 
그 외: is_plariarism = 0 


기준 모델 (NN) 은 내가 이러쿵저러쿵 classification model 만들어 보겠음.

TODO LIST
1. 데이터 수집
2. abstract pair 만들기 (한사람이 해서 배포하자. 왜냐하면 randomness 때문에 각자하면 다 달라짐)
3. simlarity 방법 별, threshold 별로 실험
'''