# -*- coding: utf-8 -*-
import sys
try: sys.path.remove('/home/donghyungko/anaconda3/lib/python3.7/site-packages')
except: pass

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.manifold import TSNE
import matplotlib as mpl
from matplotlib import font_manager, rc

# 그래프에서 마이너스 폰트 꺠지는 문제에 대한 대처
mpl.rcParams['axes.unicode_minus'] = False

# 한글 문제 해결
try:
    path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    fontprop = font_manager.FontProperties(fname=path, size=18).get_name()
    rc('font', family='NanumGothicCoding')
except: pass

import multiprocessing
import time
import pandas as pd
import re
import datetime
from collections import OrderedDict
import konlpy

import jpype

from konlpy.tag import *
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

import logging
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    
from collections import namedtuple
from sklearn.linear_model import LogisticRegression

import numpy as np
from sklearn.metrics import accuracy_score


import os
from multiprocessing import Process,Queue, Pool
import functools
from threading import Thread
import queue
import pickle

from konlpy import jvm


class NLP(object):
    
    '''
    크롤링한 데이터에 대한 텍스트분석을 위한 클래스입니다.
    
    - 텍스트 클렌징, 명사 & 형태소 추출 
    - TF-IDF 행렬 반환, 키워드 추출,
    - Doc2Vec 모델 생성 및 학습,
    - Topic 모델링의 기능을 제공합니다.
    '''
    
    
    def __init__(self):
        self.twit = Okt()
        self.kkma = Kkma()
        
        # 1. 텍스트 클렌징을 위한 정규표현식
        self.regex_ls = ['\(.+?\)',
                            '\[.+?\]',
                            '\<.+?\>',
                            '◀.+?▶',
                            '[\{\}\[\]\/?.,;:|\)*~`!^\-_+<>@▲▶◆\#$%┌─┐&\\\=\(\'\"├┼┤│┬└┴┘|ⓒ]',
                            '[\t\n\r\f\v]',
                            '[0-9]+[년월분일시조억만천원]*']
        
        # 2. 불용어 제거 리스트
        self.stopword_ls = ['에서','으로','했다','하는','이다','있다','하고','있는','까지','이라고','에는',
                            '한다','지난','관련','대한','됐다','부터','된다','위해','이번','통해','대해',
                            '애게','했다고','보다','되는','에서는','있다고','한다고','하기','에도','때문',
                            '하며','하지','해야','이어','하면','따라','하면서','라며','라고','되고','단지',
                            '이라는','이나','한다는','있따는','했고','이를','있도록','있어','하게','있다며',
                            '하기로','에서도','오는','라는','이런','하겠다고','만큼','이는','덧붙였다','있을',
                            '이고','이었다','이라','있으며','있고','이며','했다며','됐다고','나타났다','한다며',
                            '하도록','있지만','된다고','되면서','그러면서','그동안','해서는','에게','밝혔다',
                            '최근', '있다는','보이','되지','정도','지난해','매년','오늘','되며','하기도',
                            '하겠다는','했다세라',
                           ]
        
        
        
    
    #####################################################
    ################## Preprocessing ####################
    #####################################################
    
    
    # 클렌징을 위해, 추가하고 싶은 정규식을 입력하는 함수
    def add_regex(self, regex):
        '''
        텍스트 클렌징을 위한 정규표현식을 추가하는 함수입니다.
        
        inputs
        regex : str, iterable : 정규표현식 '''
        
        if type(regex) == str:
            self.regex_ls.append(regex)
        
        elif type(regex) == list:
            self.regex_ls += regex
        return
   
    
    
    # 크롤링한 text에서 불필요한 부분들을 제거하는 함수입니다.
    def _clean_text(self,text):
        for regex in self.regex_ls:
            text = re.sub(regex, '', text)
        return text
    
    
    # 복수 개의 문서를 클렌징하는 함수입니다.
    def clean_doc(self, doc_ls):
        return [self._clean_text(doc) for doc in doc_ls]
    
    
    
    # 제거하고자 하는 불용어를 추가하는 함수입니다.
    def add_stopwords(self, stopwords):
        '''
        특정 domain에서 사용되는 불용어를 제거 목록에 추가하는 함수입니다.
        모든 domain에서 사용되는 general한 불용어는 self.stopword_ls에 미리 저장되어 있습니다.
        
        inputs
        stopwords: str, iterable'''
        
        if type(stopwords) == str:
            stopwords = [stopwords]
            
        self.stopword_ls += stopwords
        return
    
    # 제거하고자 하는 불용어를 목록에서 제거하는 함수입니다.
    def delete_stopwords(self, stopwords):
        '''
        등록된 불용어 가운데, 제거하고 싶은 불용어를 추가하는 함수입니다.
        
        inputs
        stopword : str, iterable'''
        
        if type(stopwords) == str:
            stopwords = [stopwords]
        
        self.stopword_ls = [word for word in self.stopword_ls if not word in stopwords] 
        return
    
    
    # 불용어를 제거하는 함수입니다.
    def remove_stopwords(self, token_doc_ls):        
        '''
        불용어를 제거하는 함수입니다.
        
        input
        token_doc_ls : iterable, token 형태로 구분된 문서가 담긴 list 형식'''
        
        total_stopword_set = set(self.stopword_ls)
        
        # input이 복수 개의 문서가 담긴 list라면, 개별 문서에 따라 단어를 구분하여 불용어 처리
        return_ls = []
        
        if type(token_doc_ls[0]) == list:    
            for doc in token_doc_ls:
                return_ls += [[token for token in doc if not token in total_stopword_set]]
        
        elif type(token_doc_ls[0]) == str:
            return_ls = [token for token in token_doc_ls if not token in total_stopword_set]
        
        return return_ls
    
    
    # 한 개의 문서에서 명사(noun)만 추출하는 함수
    def _extract_nouns_for_single_doc(self, doc):
        clean_doc = self._clean_text(doc) # 클렌징   
        token_ls = [x for x in self.twit.nouns(clean_doc) if len(x) > 1] # 토크나이징
        return self.remove_stopwords(token_ls) # 불용어 제거
    
    
    # 한 개의 문서에서 형태소(morphs)만 추출하는 함수
    def _extract_morphs_for_single_doc(self, doc):
        clean_doc = self._clean_text(doc) # 클렌징    
        token_ls = [x for x in self.twit.morphs(clean_doc) if len(x) > 1] # 토크나이징
        return self.remove_stopwords(token_ls) # 불용어 제거
    
    
    
    # 모든 문서에서 명사(nouns)을 추출하는 함수.
    def extract_nouns_for_all_document(self,doc_ls, stopword_ls = []):
        '''
        모든 문서에서 명사를 추출하는 함수입니다.
        전처리를 적용하고 불용어를 제거한 결과를 반환합니다.
        
        input
        doc_ls : iterable, 원문이 str형태로 저장된 list
        
        return
        전처리 적용, 불용어 제거
        list : 각각의 문서가 토크나이징 된 결과를 list형태로 반환
        '''
        jpype.attachThreadToJVM()
        # 전처리
        clean_doc_ls = self.clean_doc(doc_ls)
        
        # 명사 추출
        token_doc_ls = [self._extract_nouns_for_single_doc(doc) for doc in clean_doc_ls]
        
        # 불용어 제거
        return self.remove_stopwords(token_doc_ls, stopword_ls)

    
    
    # 모든 문서에서 형태소(morph)를 추출하는 함수.
    def extract_morphs_for_all_document(self,doc_ls, stopword_ls = []):
        '''
        모든 문서에서 형태소(morph)를 추출하는 함수입니다.
        전처리를 적용하고 불용어를 제거한 결과를 반환합니다.
        
        input
        doc_ls : iterable, 원문이 str형태로 저장된 list
        
        return
        list : 각각의 문서가 토크나이징 된 결과를 list형태로 반환
        '''
        jpype.attachThreadToJVM()
        # 전처리
        clean_doc_ls = self.clean_doc(doc_ls)
        
        # 형태소(morph) 추출
        token_doc_ls = [self._extract_morphs_for_single_doc(doc) for doc in clean_doc_ls]
        
        # 불용어 제거
        return self.remove_stopwords(token_doc_ls, stopword_ls)

    
    def _extract_nouns_for_multiprocessing(self, tuple_ls):
        jpype.attachThreadToJVM()
        # 멀티프로세싱의 경우, 병렬처리시 순서가 뒤섞이는 것을 방지하기위해,
        # [(idx, doc)] 형태의 tuple이 들어온다.
        return [(idx, self._extract_nouns_for_single_doc(doc)) for idx, doc in tuple_ls]
        
        
    def _extract_morphs_for_multiprocessing(self, tuple_ls):
        jpype.attachThreadToJVM()
        # 멀티프로세싱의 경우, 병렬처리시 순서가 뒤섞이는 것을 방지하기위해,
        # [(idx, doc)] 형태의 tuple이 들어온다.
        return [(idx, self._extract_morphs_for_single_doc(doc)) for idx, doc in tuple_ls]
                  
        
                  
        
    def extract_morphs_for_all_document_FAST_VERSION(self, 
                                                     doc_ls, 
                                                     n_thread = 4):
        jpype.attachThreadToJVM()

        '''
        멀티쓰레딩을 적용하여 속도가 개선된 버전입니다.
        문서들을 전처리하고 불용어(stopwords)를 제거한 후, Tokenzing하는 함수입니다.
        
        inputs
        doc_ls : iterable, 원문이 str 형태로 담겨있는 list를 input으로 받습니다.
        n_thread: int[default : 4], 사용하실 쓰레드의 갯수를 input으로 받습니다. 
        '''
                
        # 텍스트 클렌징 작업 수행
        # [(idx, clean_doc)] 형태로 저장 (나중에 sorting을 위해)
        clean_tuple_ls = [(idx, clean_doc) for idx, clean_doc in zip(range(len(doc_ls)), self.clean_doc(doc_ls))]
        
        # 멀티쓰레딩을 위한 작업(리스트)분할
        length = len(clean_tuple_ls)
        splited_clean_tuple_ls = self.split_list(clean_tuple_ls, length//n_thread)
        
        que = queue.Queue()
        thread_ls = []
        
        for tuple_ls in splited_clean_tuple_ls:
            
            temp_thread = Thread(target= lambda q, arg1: q.put(self._extract_morphs_for_multiprocessing(arg1)),  args = (que, tuple_ls))
            
            temp_thread.start()
            thread_ls.append(temp_thread)

        for thread in thread_ls:
            thread.join()
        
        # 정렬을 위한 index_ls와 token_ls를 사용
        index_ls = []
        token_ls = []
        
        # thread의 return 값을 결합
        while not que.empty():
            
            result = que.get() # [(idx, token), (idx, token)...] 형태를 반환
            index_ls += [idx for idx, _ in result]
            token_ls += [token for _, token in result]
                       
        return [token for idx, token in sorted(zip(index_ls, token_ls))]
    

    
    
    def split_list(self, ls, n):
        '''
        병렬처리를 위해, 리스트를 원하는 크기(n)로 분할해주는 함수입니다.
        '''
        result_ls = []
        
        for i in range(0, len(ls), n):
            result_ls += [ls[i:i+n]]
        return result_ls    
    
    
    def undersample_batch(self, X_ls, Y_ls, size):
        '''
        Class Imbalance를 해결하기 위해, undersampling으로 데이터를 추출해주는 함수입니다.
        입력한 size 수에 맞춰 undersampling을 수행하며,
        k가 가장 적은 label의 수보다 큰 경우, 크기가 가장 작은 label의 수에 맞춰 undersampling을 수행합니다.
        
        inputs
        X_ls : iterable : Feature
        Y_ls : iterable : Label
        size : int : unersample size for each label
        '''
        if not type(X_ls) == list:
            try: X_ls.tolist()
            except: pass
            
        if size == len(X_ls):
            return X_ls,Y_ls
        
        unique_y_ls = list(set(Y_ls))
        
        category_dict = {}
        for x, y in zip(X_ls,Y_ls):
            try: category_dict[y] += [x]
            except: category_dict[y] = [x]
        
        
        # 전체 label의 중, 가장 적은 수
        k = np.min([len(category_dict[key]) for key in unique_y_ls])
        
        if k >= size : k = size
        else : pass
        
        batch_X_ls = []
        batch_y_ls = []
        
        # undersampling
        for key in category_dict.keys():
            batch_X_ls += category_dict[key][:k]
            batch_y_ls += [key] * k
                
        return batch_X_ls, batch_y_ls
        
        
    def oversample_batch(self, X_ls, Y_ls, size):        
        '''
        Class Imbalance를 해결하기 위해, oversampling 으로 데이터를 추출해주는 함수입니다.
        입력한 size보다 부족한 수만큼, random sampling with replacement로 oversampling을 수행합니다.
        
        inputs
        X_ls : iterable : Feature
        Y_ls : iterable : Label
        size : int : unersample size for each label
        '''
        
        if not type(X_ls) == list:
            try: X_ls.tolist()
            except: pass
            
        unique_y_ls = list(set(Y_ls))

        category_dict = {}
        for x, y in zip(X_ls,Y_ls):
            try: category_dict[y] += [x]
            except: category_dict[y] = [x]
        
        
        # 한 섹션별로 size개씩 뽑아서 하나의 batch를 만든다. (oversampling)
        
        batch_X_ls = []
        batch_y_ls = []
        
        # 표본의 수가 충분하면 size개만 추출, 부족하면 oversampling
        for key in unique_y_ls:
            if len(category_dict[key]) >= size:
                batch_X_ls += category_dict[key][:size]
                batch_y_ls += [key] * size
            
            # 부족한 수 만큼 oversample
            else:
                oversample_size = size - len(category_dict[key])

                batch_X_ls += category_dict[key]
                batch_X_ls += list(np.random.choice(category_dict[key], oversample_size, replace = True))
                
                batch_y_ls += [key] * size
                
        return batch_X_ls, batch_y_ls
    
    
    
    #####################################################
    ###################   TF-IDF   ######################
    #####################################################
        
    def doc_to_tfidf_df(self, doc_ls, 
                        min_df = 2, 
                        max_df = 0.3,
                        max_features = 50000, 
                        if_tokenized = True,
                        if_morphs = True,
                       ):
        
        '''
        각 문서에 대한 TF-IDF vector를 pandas_DataFrame으로 반환하는 함수입니다.
        
        Inputs
         - doc_ls : iterable, 
             list of documents
         
         - min_df : int or float, 
             minimum occurance in a doc (at least bigger than min_df) if float, represents minimum ratio 
         
         - max_df : int or float, 
             maximum occurange in a doc (at best smaller than max_df)
         
         - if_tokenized : Boolean, 
             True if input document is tokenized [default = True]
         
         - if_morphs : Boolean, 
             True : if not tokenized, tokenized with morphs,
             False : if not tokenized, tokenized with nouns.
        
        Return
         - TF-IDF matrix (pandas.DataFrame)
        '''
        
        if if_tokenized:
            tokenized_doc_ls = doc_ls
        else:
            if if_morphs :
                tokenized_doc_ls = self.extract_tokens_for_all_document_FAST_VERSION(doc_ls, if_morphs = True)
            else :
                tokenized_doc_ls = self.extract_tokens_for_all_document_FAST_VERSION(doc_ls, if_morphs = False)

        corpus_for_tfidf_ls = [' '.join(x) for x in tokenized_doc_ls]
        
        tfidf_vectorizer = TfidfVectorizer(max_df = max_df,
                                           min_df = min_df,
                                           max_features = max_features).fit(corpus_for_tfidf_ls)
        
        tfidf_array = tfidf_vectorizer.transform(corpus_for_tfidf_ls).toarray()
        vocab = tfidf_vectorizer.get_feature_names()
        
        self.tfidf_df = pd.DataFrame(tfidf_array, columns = vocab)
        return self.tfidf_df
    
    
    def keyword_ls_from_tfidf_df(self):
        
        ''' 
        doc_to_tfidf_df 함수를 실행한 후, 사용 가능합니다.
        TF-IDF 행렬을 기반으로, 1,2,3순위 키워드를 반환합니다.
        '''
        
        # 각 documents 별로, 1순위 키워드가 담긴 list
        first_keyword_ls = self.tfidf_df.idxmax(axis=1)
        
        # 각 documents 별로, 2순위 키워드가 담긴 list
        #second_keyword_ls = [row.sort_values(ascending =False).index[1] for index, row in self.tfidf_df.iterrows()]
        #third_keyword_ls = [row.sort_values(ascending =False).index[2] for index, row in self.tfidf_df.iterrows()]
        
        keyword_ls = [row.sort_values(ascending =False).index[0:2] for index, row in self.tfidf_df.iterrows()]
        #second_keyword_ls.append(self.tfidf_df.iloc[-1,:].sort_values(ascending =False).index[1])
        #third_keyword_ls.append(self.tfidf_df.iloc[-1,:].sort_values(ascending =False).index[2])
        
        return zip(first_keyword_ls, second_keyword_ls, third_keyword_ls)
    

    
    
    
    #####################################################
    ##################   Doc2Vec   ######################
    #####################################################
    
    
    def make_Doc2Vec_model(self,
                           dm = 1,
                           dbow_words = 0, 
                           window = 15,
                           vector_size = 300,
                           sample = 1e-5,
                           min_count = 5,
                           hs = 0,
                           negative = 5,
                           dm_mean = 0,
                           dm_concat = 0):
        '''
        Doc2Vec 모델의 초기설정을 입력하는 함수입니다.
        기존에 만들어진 모델을 load하여 사용할 수 있습니다. (load_Doc2Vec_model 함수를 사용)
        
        Inputs
         - dm : PV-DBOW / default 1
         - dbow_words : w2v simultaneous with DBOW d2v / default 0
         - window : distance between the predicted word and context words
         - vector size : vector_size
         - min_count : ignore with freq lower
         - workers : multi cpu
         - hs : hierarchical softmax / default 0
         - negative : negative sampling / default 5
        
        Return
         - None
        '''
        cores = multiprocessing.cpu_count()
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        
        self.Doc2Vec_model = Doc2Vec(
            dm= dm,                     # PV-DBOW / default 1
            dbow_words= dbow_words,     # w2v simultaneous with DBOW d2v / default 0
            window= window,             # distance between the predicted word and context words
            vector_size= vector_size,   # vector size
            sample = sample,
            min_count = min_count,      # ignore with freq lower
            workers= cores,             # multi cpu
            hs = hs,                    # hierarchical softmax / default 0
            negative = negative,        # negative sampling / default 5
            dm_mean = dm_mean,
            dm_concat = dm_concat,
        )
        
        return
            
        
        
    def build_and_train_Doc2Vec_model(self,
                                      train_doc_ls,
                                      train_tag_ls,
                                      n_epochs = 10,
                                      if_tokenized = True,
                                      if_morphs = True):
        
        ''' 
        Doc2Vec 모델을 생성 혹은 Load한 다음 작업으로, Doc2Vec을 build하고 학습을 수행합니다.
        
        Inputs
         - train_doc_ls : iterable, documents(tokenized or not tokenized)
         - train_tag_ls : iterable, tags of each documents
         - n_epochs : int, numbers of iteration
         - if_morphs : 원문에 대한 tokenizing을 수행할 때, morphs를 추출 (defaulf = True),
         - if_tokenized : Boolean, True if input document is tokenized [default = True]
         - if_morphs : Boolean, 
                       True : if not tokenized, tokenized with morphs,
                       False : if not tokenized, tokenized with nouns.
        
        Return
         - None
        '''
        
                
        # tokenized된 input데이터를 받으면 tokenizing skip
        if if_tokenized :
            train_token_ls = train_doc_ls
        else:
            clean_train_doc_ls = self.clean_doc(train_doc_ls)
            if if_morphs:
                train_token_ls = self.extract_tokens_for_all_document_FAST_VERSION(clean_train_doc_ls, if_morphs = True)
            else:
                train_token_ls = self.extract_tokens_for_all_document_FAST_VERSION(clean_train_doc_ls, if_morphs = False)
        
        # train_tag_ls를 리스트 형태로 변환 (Series 형태로 들어올 경우)
        try:  train_tag_ls = train_tag_ls.tolist()
        except:   pass
            
        # words와 tags로 구성된 namedtuple 형태로 데이터 변환 (tagging 작업)
        tagged_train_doc_ls = [TaggedDocument(tuple_[0], [tuple_[1]]) for i, tuple_ in enumerate(zip(train_token_ls, train_tag_ls))]
        
        # Doc2Vec 모델에 단어 build작업 수행
        self.Doc2Vec_model.build_vocab(tagged_train_doc_ls)

        # 학습 수행
        self.Doc2Vec_model.train(tagged_train_doc_ls,
                                 total_examples= self.Doc2Vec_model.corpus_count,
                                 epochs= n_epochs)
        return
    
    
    def train_Doc2Vec_model(self,
                            train_doc_ls,
                            train_tag_ls,
                            n_epochs = 10,
                            if_tokenized = True,
                            if_morphs = True,
                            ):
        ''' 
        built된 Doc2Vec 모델에 추가적인 학습을 수행합니다.
        
        Inputs
         - train_doc_ls : iterable, documents(tokenized or not tokenized)
         - train_tag_ls : iterable, tags of each documents
         - n_epochs : int, numbers of iteration
         - if_morphs : 원문에 대한 tokenizing을 수행할 때, morphs를 추출 (defaulf = True),
         - if_tokenized : Boolean, True if input document is tokenized [default = True]
         - if_morphs : Boolean, 
                       True : if not tokenized, tokenized with morphs,
                       False : if not tokenized, tokenized with nouns.
        
        Return
         - None
        '''
        
                
        # tokenized된 input데이터를 받으면 tokenizing skip
        if if_tokenized :
            train_token_ls = train_doc_ls
        else:
            clean_train_doc_ls = self.clean_doc(train_doc_ls)
            if if_morphs:
                train_token_ls = self.extract_tokens_for_all_document_FAST_VERSION(clean_train_doc_ls, if_morphs = True)
            else :
                train_token_ls = self.extract_tokens_for_all_document_FAST_VERSION(clean_train_doc_ls, if_morphs = False)
        
        # train_tag_ls를 리스트 형태로 변환 (Series 형태로 들어올 경우)
        try:  train_tag_ls = train_tag_ls.tolist()
        except:   pass
            
        # words와 tags로 구성된 namedtuple 형태로 데이터 변환 (tagging 작업)
        tagged_train_doc_ls = [TaggedDocument(tuple_[0], [tuple_[1]]) for i, tuple_ in enumerate(zip(train_token_ls, train_tag_ls))]
        
        # 학습 수행
        self.Doc2Vec_model.train(tagged_train_doc_ls,
                                 total_examples= self.Doc2Vec_model.corpus_count,
                                 epochs= n_epochs)
        return
    
    
    
    def infer_vectors_with_Doc2Vec(self,doc_ls, 
                                   alpha = 0.1,
                                   steps = 30):
        
        '''
        Doc2Vec을 사용하여, documents를 vectorize하는 함수입니다.
        
        Inputs
         - doc_ls : iterable or str, array of tokenized documents
         
        return
         - matrix of documents inferred by Doc2Vec
        '''
        
        return_ls = []
        
        # 문서 1개가 들어온 경우,
        if type(doc_ls) == str:
            return self.Doc2Vec_model.infer_vector(doc, 
                                                   alpha = alpha, 
                                                   min_alpha = self.Doc2Vec_model.min_alpha,
                                                   steps = steps)
        
        # 복수 개의 문서가 input으로 들어온 경우,
        else:
            return [self.Doc2Vec_model.infer_vector(doc,
                                                    alpha = alpha,
                                                    min_alpha = self.Doc2Vec_model.min_alpha,
                                                    steps = steps) \
                    for doc in doc_ls]
        
        
    def load_Doc2Vec_model(self, model_name):
        self.Doc2Vec_model = Doc2Vec.load(model_name)
        return  self.Doc2Vec_model
    
    

