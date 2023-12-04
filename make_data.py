import json
from utils import *
from nltk import sent_tokenize
from tqdm import tqdm

fnames = ['DS', 'NRF', 'VC']

for fname in fnames:
    with open(f'./data/{fname}.json', 'r') as f:
        data = json.load(f)

    augmented_data = []

    THRESHOLD = 0.5

    for i in tqdm(range(len(data))):
        abstract = data[i]['content']
        title = data[i]['title']
        # Split the abstract into sentences as a list
        paragraph = sent_tokenize(abstract)
        data_id = fname + '_' + str(i)
        '''
        In the utils.py, we have various functions to augment the data
        synonym_replacement: replace words with their synonyms
        random_insertion: insert random synonyms into the sentence
        random_swap: swap words randomly
        random_deletion: delete words randomly
        paraphrase: paraphrase uses the parrot model to paraphrase the sentence
        '''
        
        # original data
        augmented_data.append({'id': str(data_id), 'title': title, 'content': abstract})

        # For example, if you want to use synonym_replacement, you can do the following:
        # rate is the rate of replacement. 0.1 means 10% of the words will be replaced in the sentence
        

        # SR
        modified_abstracts = synonym_replacement(paragraph, rate = 0.2, num_aug = 1)
        augmented_data.append({'id': data_id+'_SR', 'title': title, 'content': modified_abstracts[0]})

        # RI
        modified_abstracts = random_insertion(paragraph, rate = 0.2, num_aug = 1)
        augmented_data.append({'id': data_id+'_RI', 'title': title, 'content': modified_abstracts[0]})

        # RS
        modified_abstracts = random_swap(paragraph, rate = 0.2, num_aug = 1)
        augmented_data.append({'id': data_id+'_RS', 'title': title, 'content': modified_abstracts[0]})

        # RD
        modified_abstracts = random_deletion(paragraph, rate = 0.2, num_aug = 1)
        augmented_data.append({'id': data_id+'_RD', 'title': title, 'content': modified_abstracts[0]})

        # MIX
        modified_abstracts = mix_eda(paragraph, rate = 0.1, num_aug = 1)
        augmented_data.append({'id': data_id+'_MIX', 'title': title, 'content': modified_abstracts[0]})

        # PP
        modified_abstracts = paraphrase(paragraph, adequacy_thres = 0.9, fluency_thres = 0.9, num_aug = 1)
        augmented_data.append({'id': data_id+'_PP', 'title': title, 'content': modified_abstracts[0]})

    

    new_fname = fname + '_aug.json'
    with open(f'./data/{new_fname}', 'w') as f:
        json.dump(augmented_data, f)


with open('./data/DS_aug.json', 'r') as f:
    ds = json.load(f)

with open('./data/NRF_aug.json', 'r') as f:
    nrf = json.load(f)

with open('./data/VC_aug.json', 'r') as f:
    vc = json.load(f)

import random
from itertools import combinations

data = []
used = set([])
for i in range(0, len(ds), 7):
    title = ds[i]['title']
    perms = list(combinations(range(i, i+7), 2))
    for j, k in perms:
        new_id = ds[j]['id']+'_'+ds[k]['id']
        other_id = ds[k]['id']+'_'+ds[j]['id']
        if new_id in used or other_id in used:
            continue
        used.add(new_id)
        used.add(other_id)
        data.append({'id': str(new_id), 'title1': title, 'title2': title, 'content1': ds[j]['content'], 'content2': ds[k]['content'], 'is_plagiarism': 1})

    candidates = [l for l in range(len(ds)) if l not in range(i, i+7) and l not in used]
    rand_nums = random.sample(candidates, 10)
    n = 0
    while n < 10:
        j = random.sample(candidates, 1)[0]
        new_id = ds[i]['id']+'_'+ds[j]['id']
        other_id = ds[j]['id']+'_'+ds[i]['id']
        if new_id in used or other_id in used:
            continue
        data.append({'id': str(new_id), 'title1': ds[i]['title'], 'title2': ds[j]['title'], 'content1': ds[i]['content'], 'content2': ds[j]['content'], 'is_plagiarism': 0})
        used.add(new_id)
        used.add(other_id)
        n += 1
    n = 0
    while n < 5:
        j = random.sample(candidates, 1)[0]
        new_id = ds[i]['id'] + '_' + nrf[j]['id']
        other_id = nrf[j]['id'] + '_' + ds[i]['id']
        if new_id in used or other_id in used:
            continue
        data.append({'id': str(new_id),'title1': ds[i]['title'], 'title2': nrf[j]['title'], 'content1': ds[i]['content'], 'content2': nrf[j]['content'], 'is_plagiarism': 0})
        used.add(new_id)
        used.add(other_id)
        n += 1
    n = 0
    while n < 6:
        j = random.sample(candidates, 1)[0]
        new_id = ds[i]['id'] + '_' + vc[j]['id']
        other_id = vc[j]['id'] + '_' + ds[i]['id']
        if new_id in used or other_id in used:
            continue
        data.append({'id': str(new_id),'title1': ds[i]['title'], 'title2': vc[j]['title'], 'content1': ds[i]['content'], 'content2': vc[j]['content'], 'is_plagiarism': 0})
        used.add(new_id)
        used.add(other_id)
        n += 1

for i in range(0, len(nrf), 7):
    title = nrf[i]['title']
    perms = list(combinations(range(i, i+7), 2))
    for j, k in perms:
        new_id = nrf[j]['id']+'_'+nrf[k]['id']
        other_id = nrf[k]['id']+'_'+nrf[j]['id']
        if new_id in used or other_id in used:
            continue
        used.add(new_id)
        used.add(other_id)
        data.append({'id': str(new_id), 'title1': title, 'title2': title, 'content1': nrf[j]['content'], 'content2': nrf[k]['content'], 'is_plagiarism': 1})

    candidates = [l for l in range(len(nrf)) if l not in range(i, i+7) and l not in used]
    rand_nums = random.sample(candidates, 10)
    n = 0
    while n < 10:
        j = random.sample(candidates, 1)[0]
        new_id = nrf[i]['id']+'_'+nrf[j]['id']
        other_id = nrf[j]['id']+'_'+nrf[i]['id']
        if new_id in used or other_id in used:
            continue
        data.append({'id': str(new_id), 'title1': nrf[i]['title'], 'title2': nrf[j]['title'], 'content1': nrf[i]['content'], 'content2': nrf[j]['content'], 'is_plagiarism': 0})
        used.add(new_id)
        used.add(other_id)
        n += 1
    n = 0
    while n < 5:
        j = random.sample(candidates, 1)[0]
        new_id = nrf[i]['id'] + '_' + ds[j]['id']
        other_id = ds[j]['id'] + '_' + nrf[i]['id']
        if new_id in used or other_id in used:
            continue
        data.append({'id': str(new_id),'title1': nrf[i]['title'], 'title2': ds[j]['title'], 'content1': nrf[i]['content'], 'content2': ds[j]['content'], 'is_plagiarism': 0})
        used.add(new_id)
        used.add(other_id)
        n += 1
    n = 0
    while n < 6:
        j = random.sample(candidates, 1)[0]
        new_id = nrf[i]['id'] + '_' + vc[j]['id']
        other_id = vc[j]['id'] + '_' + nrf[i]['id']
        if new_id in used or other_id in used:
            continue
        data.append({'id': str(new_id),'title1': nrf[i]['title'], 'title2': vc[j]['title'], 'content1': nrf[i]['content'], 'content2': vc[j]['content'], 'is_plagiarism': 0})
        used.add(new_id)
        used.add(other_id)
        n += 1

for i in range(0, len(vc), 7):
    title = vc[i]['title']
    perms = list(combinations(range(i, i+7), 2))
    for j, k in perms:
        new_id = vc[j]['id']+'_'+vc[k]['id']
        other_id = vc[k]['id']+'_'+vc[j]['id']
        if new_id in used or other_id in used:
            continue
        used.add(new_id)
        used.add(other_id)
        data.append({'id': str(new_id), 'title1': title, 'title2': title, 'content1': vc[j]['content'], 'content2': vc[k]['content'], 'is_plagiarism': 1})

    candidates = [l for l in range(len(vc)) if l not in range(i, i+7) and l not in used]
    rand_nums = random.sample(candidates, 10)
    n = 0
    while n < 10:
        j = random.sample(candidates, 1)[0]
        new_id = vc[i]['id']+'_'+vc[j]['id']
        other_id = vc[j]['id']+'_'+vc[i]['id']
        if new_id in used or other_id in used:
            continue
        data.append({'id': str(new_id), 'title1': vc[i]['title'], 'title2': vc[j]['title'], 'content1': vc[i]['content'], 'content2': vc[j]['content'], 'is_plagiarism': 0})
        used.add(new_id)
        used.add(other_id)
        n += 1
    n = 0
    while n < 5:
        j = random.sample(candidates, 1)[0]
        new_id = vc[i]['id'] + '_' + ds[j]['id']
        other_id = ds[j]['id'] + '_' + vc[i]['id']
        if new_id in used or other_id in used:
            continue
        data.append({'id': str(new_id),'title1': vc[i]['title'], 'title2': ds[j]['title'], 'content1': vc[i]['content'], 'content2': ds[j]['content'], 'is_plagiarism': 0})
        used.add(new_id)
        used.add(other_id)
        n += 1
    n = 0
    while n < 6:
        j = random.sample(candidates, 1)[0]
        new_id = vc[i]['id'] + '_' + nrf[j]['id']
        other_id = nrf[j]['id'] + '_' + vc[i]['id']
        if new_id in used or other_id in used:
            continue
        data.append({'id': str(new_id),'title1': vc[i]['title'], 'title2': nrf[j]['title'], 'content1': vc[i]['content'], 'content2': nrf[j]['content'], 'is_plagiarism': 0})
        used.add(new_id)
        used.add(other_id)
        n += 1
random.shuffle(data)
with open('./data/data.json', 'w') as f:
    json.dump(data, f, indent = 4, ensure_ascii = False)

with open('./data/data.json', 'r') as f:
    data = json.load(f)
train = data[:int(len(data)*0.9)]
test = data[int(len(data)*0.9):]

with open('./data/train.json', 'w') as f:
    json.dump(train, f, indent = 4, ensure_ascii = False)

with open('./data/test.json', 'w') as f:
    json.dump(test, f, indent = 4, ensure_ascii = False)

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
1. 데이터 수집 -> Done
2. abstract pair 만들기 (한사람이 해서 배포하자. 왜냐하면 randomness 때문에 각자하면 다 달라짐) -> Done
3. simlarity 방법 별, threshold 별로 실험
'''