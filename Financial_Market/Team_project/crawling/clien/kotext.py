import io
import re
import pandas as pd
 
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage
 
from konlpy.tag import *
import konlpy
from gensim.models import word2vec
from collections import Counter

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import gensim
import matplotlib as mpl
import seaborn as sns
sns.set()
from matplotlib import font_manager, rc
from PIL import Image

font_location = 'C:/Windows/Fonts/malgun.ttf' # For Windows
font_name = font_manager.FontProperties(fname=font_location).get_name()
rc('font', family=font_name)


# 그래프에서 마이너스 폰트 꺠지는 문제에 대한 대처
mpl.rcParams['axes.unicode_minus'] = False


def extract_text_from_pdf(pdf_path):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle)
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
 
    with open(pdf_path, 'rb') as fh:
        for page in PDFPage.get_pages(fh, 
                                      caching=True,
                                      check_extractable=True):
            page_interpreter.process_page(page)
 
        text = fake_file_handle.getvalue()
 
    # close open handles
    converter.close()
    fake_file_handle.close()
 
    if text:
        return text
    
def remove_words(word_ls,text_ls):
    
    for word in word_ls:
        text_ls = [text.replace(word,'') for text in text_ls]
    
    return text_ls



def extract_nouns(text_ls):
    twit = Twitter()

    temp_nouns = twit.nouns(' '.join(text_ls))
    nouns = [x for x in temp_nouns if len(x) > 1]

    print('총 단어 수 :', len(nouns))
    print('중복을 제거한 단어의 수 :', len(set(nouns)))
    
    return nouns




def extract_morphs(text_ls):
    twit = Twitter()
    temp_morphs = twit.morphs(' '.join(text_ls))
    morphs = [x for x in temp_morphs if len(x) > 1]

    print('총 단어 수 :',len(morphs))
    print('중복을 제거한 단어의 수 :', len(set(morphs)))
    
    return morphs




def make_word_cloud(word_ls, fontsize = 50, figsize = (12,8), max_words = 1000, image_path = 'house.png'):
    
    font_path = '/usr/share/fonts/truetype/nanum/NanumMyeongjoBold.ttf'
    plt.figure(figsize = figsize)

    img_mask = np.array(Image.open(image_path))

    word_cloud = WordCloud(background_color = 'white',
                           font_path = font_path,
                           max_words = 1000,
                           max_font_size= fontsize,
                           mask = img_mask,
                          )

    wc = word_cloud.generate(' '.join(word_ls))

    plt.imshow(wc)
    plt.axis('off')
    plt.show()
    
    
    
    
def preprocessing_for_embedding(text_ls):
        
    total_ls = []
    
    # ls는 하나의 게시글을 개별 값으로 담고있는 리스트이다.
    twit = Twitter()
    
    for text in text_ls:
        noun_ls = twit.nouns(text)
        
        total_ls += [[x for x in noun_ls if len(x) > 1]]
        
    return total_ls




def embedding(sentences, num_features = 300, min_word_count = 50, num_workers = 4, context = 10,):
    
    model_name = "%sfeatures_%sminwords_%scontext"%(num_features, min_word_count, context)
    
    try:   
        model = gensim.models.Doc2Vec.load(model_name)
    
    except:
        # 파라미터 값 지정
        '''
        num_features = 300 # 문자 벡터 차원 수
        min_word_count = 50 # 최소 문자 수
        num_workers =4 # 병렬 처리 스레드 수
        context = 10 # 문자열의 창 크기
        '''
        downsampling = 1e-3 # 문자 빈도 수 Downsample


        model = word2vec.Word2Vec(sentences,
                                  workers = num_workers,
                                  size = num_features,
                                  min_count = min_word_count,
                                  window = context,
                                  sample = downsampling,
                                  iter = 30
                                 )

    # If you don't plan to train the model any further, calling 
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and 
    # save the model for later use. You can load it later using Word2Vec.load()
    model.save(model_name)
    
    return model, model_name




def make_embedding_plot(model,model_name = ''):
    
    if model_name != '':
        model = gensim.models.Doc2Vec.load(model_name)
    
    vocab = [x for x in model.wv.vocab]
    
    X = model[vocab]

    # n_components는 차원의 수
    tsne = TSNE(n_components = 2)

    X_tsne = tsne.fit_transform(X)
    
    df = pd.DataFrame(X_tsne, index = vocab, columns = ['x','y'])
    
    plt.figure(figsize = (30,30))

    plt.scatter(df['x'].values, df['y'].values)

    for words, pos in df.iterrows():
        plt.annotate(words, pos, fontsize = 15)
        