import json



with open('./data/DS_aug.json', 'r') as f:
    nrf = json.load(f)

with open('./data/NRF_aug.json', 'r') as f:
    vc = json.load(f)

with open('./data/VC_aug.json', 'r') as f:
    ds = json.load(f)

import random
from itertools import combinations

data = []
used = set([])


for i in range(0, len(vc), 7):
    title = vc[i]['title']
    perms = list(combinations(range(i, i+6), 2))
    for j, k in perms:
        if len(vc[k]['id'].split('_')) > 2:
            new_id = vc[j]['id']+'_'+vc[k]['id'].split('_')[1]+vc[k]['id'].split('_')[2]
        else:
            new_id = vc[j]['id']+'_'+vc[k]['id'].split('_')[-1]
        if len(vc[j]['id'].split('_')) > 2:
            other_id = vc[k]['id']+'_'+vc[j]['id'].split('_')[1]+vc[j]['id'].split('_')[2]
        else:
            other_id = vc[k]['id']+'_'+vc[j]['id'].split('_')[-1]
        if new_id in used or other_id in used:
            continue
        used.add(new_id)
        used.add(other_id)
        data.append({'id': str(new_id), 'title1': title, 'title2': title, 'content1': vc[j]['content'], 'content2': vc[k]['content'], 'is_plagiarism': 1})

    candidates = [l for l in range(len(vc)) if l not in range(i, i+6) and l not in used]
    rand_nums = random.sample(candidates, 10)
    n = 0
    while n < 10:
        j = random.sample(candidates, 1)[0]
        if len(vc[j]['id'].split('_')) > 2:
            new_id = vc[i]['id']+'_'+vc[j]['id'].split('_')[1]+vc[j]['id'].split('_')[2]
        else:
            new_id = vc[i]['id']+'_'+vc[j]['id'].split('_')[-1]
        other_id = vc[j]['id']+'_'+vc[i]['id'].split('_')[-1]
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
    while n < 5:
        j = random.sample(candidates, 1)[0]
        new_id = vc[i]['id'] + '_' + nrf[j]['id']
        other_id = nrf[j]['id'] + '_' + vc[i]['id']
        if new_id in used or other_id in used:
            continue
        data.append({'id': str(new_id),'title1': vc[i]['title'], 'title2': nrf[j]['title'], 'content1': vc[i]['content'], 'content2': nrf[j]['content'], 'is_plagiarism': 0})
        used.add(new_id)
        used.add(other_id)
        n += 1