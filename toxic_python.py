#!/usr/bin/env python
# coding: utf-8

# In[95]:


import pandas as pd
import numpy as np
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
import re

w = pd.read_csv("train.csv")


# In[100]:


urgent_rows = w.iloc[:, 2] == 1 ### Getting Urgent Mails # Find a way to create id

urgent_text = w[urgent_rows]
urgent_text = urgent_text.iloc[: , 1]

urgent_text = urgent_text.iloc[range(1000)]


# In[83]:


### urgent_text = str(urgent_text)
### urgent_text = re.sub("[^a-zA-Z]" , " " , urgent_text).lower()

### urg = urgent_text.split()
### urg = [word for word in urg if not word in stopwords.words("english")]
### from nltk.stem.porter import PorterStemmer
### ps = PorterStemmer()


# In[106]:


a = urgent_text.apply(lambda x : x.lower()).apply(lambda x : re.sub("[^a-zA-Z]" , " " , x))


# In[107]:


b = str(a.iloc[0])

for i in range(1 , len(a)): 
    b += a.iloc[i]

b = b.split()
b = [word for word in b if not word in stopwords.words("english")]


# In[151]:


def wordListToPB(wordlist):
    wordfreq = [wordlist.count(p) for p in wordlist]
    return dict(list(zip(wordlist,wordfreq)))

b_dict = wordListToPB(b)


# In[144]:


nonurgent_rows = w.iloc[:, 2] == 0 

nonurgent_text = w[nonurgent_rows]
nonurgent_text = nonurgent_text.iloc[: , 1]

nonurgent_text = nonurgent_text.iloc[range(1000)]


# In[146]:


c = nonurgent_text.apply(lambda x : x.lower()).apply(lambda x : re.sub("[^a-zA-Z]" , " " , x))

d = str(c.iloc[0])

for i in range(1 , len(c)): 
    d += c.iloc[i]

d = d.split()
d = [word for word in d if not word in stopwords.words("english")]


# In[149]:


d_dict = wordListToPB(d)


# In[155]:


bd = b + d


# In[163]:


def checkKey(dict, key): 
      
    if key in dict.keys(): 
        return dict[key]
    
    else: 
        return 0  


# In[188]:


bd_dict = dict()


# In[198]:


def bd_prob(string) : 
    for i in range(len(string)):
        n_urgent = checkKey(b_dict , string[i])
        n_nonurgent = checkKey(d_dict , string[i])
        
        bd_dict[string[i]] = n_urgent/(n_urgent + n_nonurgent) 
    
    return bd_dict


# In[203]:


bd_prob = bd_prob(bd)


# In[207]:


def predict(sentence):
    urgency_parameter = 0
    
    for i in range(len(sentence)) : 
        urgency_parameter += bd_prob[sentence[i]]
      
    if urgency_parameter >= 1 : 
        print("Urgent")
    
    else:
        print("Not-Urgent")
    
    return urgency_parameter


# In[205]:


y = str(a.iloc[0])
y = str(y).lower()
y = re.sub("[^a-zA-Z]" , " " , y)
y = y.split()

y = [word for word in y if not word in stopwords.words("english")]


# In[208]:


predict(y)


# In[ ]:




