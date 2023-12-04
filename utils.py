
import eda
from transformers import BertTokenizer, BertModel
from parrot import Parrot
import torch
import warnings
import numpy as np

warnings.filterwarnings("ignore")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")


def get_embedding(paragraph):   
    return model(tokenizer(paragraph, return_tensors="pt", padding=True)['input_ids'])[0].mean(axis=1).detach().numpy()


def synonym_replacement(paragraph, rate, num_aug):
    aug_paragraphs = [' ' for i in range(num_aug)]
    for sentence in paragraph:
        replaced = eda.eda(sentence, alpha_sr = rate, alpha_ri = 0, alpha_rs = 0, p_rd = 0, num_aug = num_aug, stopwords = False)
        for i in range(num_aug):
            aug_paragraphs[i] += replaced[i]
    return aug_paragraphs

def random_insertion(paragraph, rate, num_aug):
    aug_paragraphs = [' ' for i in range(num_aug)]
    for sentence in paragraph:
        replaced = eda.eda(sentence, alpha_sr = 0, alpha_ri = rate, alpha_rs = 0, p_rd = 0, num_aug = num_aug, stopwords = False)
        for i in range(num_aug):
            aug_paragraphs[i] += replaced[i]
    return aug_paragraphs

def random_swap(paragraph, rate, num_aug):
    aug_paragraphs = [' ' for i in range(num_aug)]
    for sentence in paragraph:
        replaced = eda.eda(sentence, alpha_sr = 0, alpha_ri = 0, alpha_rs = rate, p_rd = 0, num_aug = num_aug, stopwords = False)
        for i in range(num_aug):
            aug_paragraphs[i] += replaced[i]
    return aug_paragraphs

def random_deletion(paragraph, rate, num_aug):
    aug_paragraphs = [' ' for i in range(num_aug)]
    for sentence in paragraph:
        replaced = eda.eda(sentence, alpha_sr = 0, alpha_ri = 0, alpha_rs = 0, p_rd = rate, num_aug = num_aug, stopwords = False)
        for i in range(num_aug):
            aug_paragraphs[i] += replaced[i]
    return aug_paragraphs

def mix_eda(paragraph, rate, num_aug):
    aug_paragraphs = [' ' for i in range(num_aug)]
    for sentence in paragraph:
        replaced = eda.eda(sentence, alpha_sr = rate, alpha_ri = rate, alpha_rs = rate, p_rd = rate, num_aug = num_aug, stopwords = False)
        for i in range(num_aug):
            aug_paragraphs[i] += replaced[i]
    return aug_paragraphs


'''
Adequacy: How much the paraphrase retains the meaning of the original sentence?
Fluency: Is the paraphrase fluent correct?
Diversity: ow much has the paraphrase changed the original sentence?

high adquacy_threshold -> high adequacy
low fluency_threshold -> high fluency

adequacy_threshold: (float) threshold for adequacy. 0~1
fluency_threshold: (float) threshold for fluency. 0~1
Diversity: (bool) whether to use diverse paraphrasing or not.

Note that if you set the threshold too high, you may get None.

plz refer to https://github.com/PrithivirajDamodaran/Parrot_Paraphraser for more details
'''
parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5")

def paraphrase(paragraph, diversity = False, adequacy_thres = 0.9, fluency_thres = 0.9, num_aug = 3):
    aug_paragraphs = [' ' for i in range(num_aug)]
    for sentence in paragraph:
        replaced = parrot.augment(sentence, do_diverse=diversity, adequacy_threshold =adequacy_thres, fluency_threshold=fluency_thres, max_return_phrases = num_aug)
        while replaced == None:
            adequacy_thres -= 0.05
            fluency_thres -= 0.05
            replaced = parrot.augment(sentence, do_diverse=diversity, adequacy_threshold =adequacy_thres, fluency_threshold=fluency_thres, max_return_phrases = num_aug)
        for i in range(num_aug):
            aug_paragraphs[i] += replaced[i][0]
    return aug_paragraphs

def cosine_similarity(a, b):
    a = a.reshape(-1,)
    b = b.reshape(-1,)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def euclidean_distance(a, b):
    a = a.reshape(-1,)
    b = b.reshape(-1,)
    return np.linalg.norm(a-b)

def correlation_coefficient(a, b):
    return np.corrcoef(a, b)[0, 1]

def get_similarity(paragraph1, paragraph2, metric = 'cosine'):
    if metric == 'cosine':
        return cosine_similarity(get_embedding(paragraph1), get_embedding(paragraph2))
    elif metric == 'euclidean':
        return euclidean_distance(get_embedding(paragraph1), get_embedding(paragraph2))
    elif metric == 'correlation':
        return correlation_coefficient(get_embedding(paragraph1), get_embedding(paragraph2))
    else:
        raise Exception('Invalid metric. Please choose from cosine, euclidean, correlation')