# Install Library

```bash
pip install nltk

python -m nltk.downloader all

pip install git+https://github.com/PrithivirajDamodaran/Parrot_Paraphraser.git

pip install transformers

```



# Background

EDA 는 text augmentation 에 쓰이는 기법으로
synonym replacement, random insertion, random deletion, random swap 총 네가지 기법으로 구성된다. 
본 팀 프로젝트에서는 각각의 기법 (SR, RI, RD, RW) 와 함께 네가지 기법을 모두 섞은 MIX 와 parrot api를 사용하는 총 6가지 방법의 text augmentation 을 적용한다.



# Data

./data 내의 train.json, test.json

id, title1, title2, content1, content2, is_plagiarism 으로 구성

*is_plagiarism: 1 if true, 0 else

id 예시1) NRF_75_RD_NRF_75_MIX: NRF 75번째 논문의 Random Deletion으로 aug 한 abstract+ mix로 aug한 abstract -> is_plagiarism: 1
id 예시2) DS_72_DS_95: Dialogue system 72번째 논문의 original abstract와 95번째 논문의 original abstract -> is_plagiarism: 0

id 예시3) NRF_51_DS_70_RD: NRF 71번째 논문의 original abstract와 DS 70번째 논문을 Random deletion으로 augmentation 한 abstract -> is_plagiarism: 0



# How to get embedding?

Just use *get_embedding* function in *utils.py*



# Proximity Measurement

Cosine Similarity, Euclidean Distance, Correlation Coefficient

전부 다 *utils.py* 에 정의 되어 있음.