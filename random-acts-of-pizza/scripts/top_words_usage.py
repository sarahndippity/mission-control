import pandas as pd
import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import *
from sklearn import metrics


def key_words_usage(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create one new column in training dataset: percentage of key words used.
    
    Function takes in a pandas Dataframe as input, and returns the same Dataframe
    with the 1 additional column.
    """
    # prep data 
    df = train_df.copy()
    train_df_vec = df["title_and_request"]
    train_label_vec = df["requester_received_pizza"]

    # initialize vectorizer using preprocessor and stop words
    vectorizer = TfidfVectorizer(preprocessor = processor) #, stop_words='english')
    vector_train = vectorizer.fit_transform(train_df_vec)

    # run logistic regression
    lr = LogisticRegression(solver='liblinear', multi_class='auto')
    lr.fit(vector_train, train_label_vec)

    # get top 20 features
    ind = np.fliplr(np.argsort(lr.coef_, axis = 1))[:,:20]
    results = []
    top_features = []
    for j in range(len(ind[0])): 
        feature = vectorizer.get_feature_names()[ind[0,j]]
        coef_1 = lr.coef_[0,ind[0,j]]
        top_features.append(feature)
        results.append([feature, coef_1])
        
    # count number of words in post and calculate percentage of key words used
    def wordCounterUsage(text):
        count_of_key_words = 0
        for i in top_features:
            in_text = 1 if i in text else 0
            count_of_key_words += in_text
        return count_of_key_words/len(top_features)
    

    df['key_words_usage'] = [wordCounterUsage(text) for text in df['title_and_request']]
    
    
    return df, results



def processor(s):
    """
    Takes text as input and returns cleaned up text
    """
    s = s.lower()
    s = ' '.join(word for word in s.split()
                 if word not in ENGLISH_STOP_WORDS)
                         
    s = re.sub(r'ies ', 'y ', s)
    s = re.sub(r'(ed|al|ally|s|ment)$', ' ', s)
    s = re.sub(r'(ly)$', '', s)
    s = re.sub(r"won\'t", 'will not', s)
    s = re.sub(r"can\'t", 'can not', s)
    s = re.sub('cannot', 'can not', s)
    s = re.sub(r"n\'t", ' not', s)
    s = re.sub(r"\'re", ' are', s)
    s = re.sub(r"\'ll", ' will', s)
    s = re.sub(r"\'ve", ' have', s)
    s = re.sub(r"\'m", ' am', s)
    s = re.sub('[^\w\d]',' ', s)
    s = re.sub('([\d]+)', ' numbers_seq ', s)

    return s


# train_df, vector_words = key_words_usage(train_df)