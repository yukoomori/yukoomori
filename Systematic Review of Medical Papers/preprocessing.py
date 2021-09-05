#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from nltk.tokenize import word_tokenize
from tqdm import tqdm


def clean(df, col, cache):
    # fill NaN
    df = df.fillna(value = "")
    
    # tokenize
    df_1 = df
    df_1[col] = df_1[col].apply(word_tokenize) 
        
    # get rid of stopwords
    df_2 = df_1
    df_2[col] = df_2[col].apply(lambda x: [word for word in tqdm(x) if word not in cache])
        
    # remove punctionations
    df_3 = df_2
    punc = {",", ".", "?", ";", ":", "'", '"', "(", ")", "[", "]", "{", "}", "|", "/"}
    df_3[col] = df_3[col].apply(lambda x: [word for word in tqdm(x) if word not in punc])
    
    return df_3 # a clean one

