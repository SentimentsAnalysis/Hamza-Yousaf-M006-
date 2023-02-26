#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pandas')




# In[2]:


pip install neattext


# In[3]:


get_ipython().system('pip install matplotlib')


# In[4]:


import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


#import data set
from google.colab import files
upload = files.upload()


# In[5]:


#read data set 
turkey = pd.read_csv('turkey.csv', encoding = 'ISO-8859-1')
 


# In[6]:


#display first 5 rows of the data set
turkey.head()


# In[7]:


#display last 5 rows of the data set
turkey.tail()


# In[8]:


#rows and clm
turkey.shape


# In[9]:


#information about the data set
turkey.info()


# In[10]:


#calculate null values in the data set
turkey.isnull().sum()


# In[11]:


#drop the null values
turkey = turkey.dropna()


# In[12]:


#after deleting again check the null values
turkey.isnull().sum()


# In[13]:


#after the deletion of null values again check the shape of the data set
turkey.shape


# In[14]:


#import cleaning package
import neattext.functions as nfx


# In[15]:


#show total packages in neattext pacakge
dir(nfx)


# In[16]:


#showing tweet before cleaning (all tweets)
turkey['Tweet']


# In[17]:


#specific tweet (before cleaning)
turkey['Tweet'].iloc[2]


# In[18]:


#extract hashtags from tweet column
turkey['Tweet'].apply(nfx.extract_hashtags)


# In[19]:


#store extracted hashtags in new column called "extracted hashtags"
turkey['extracted_hashtags']=turkey['Tweet'].apply(nfx.extract_hashtags)


# In[20]:


#compare extracted hashtags with tweets(before cleaning)
turkey[['extracted_hashtags','Tweet']]


# In[21]:


#matching of extracted hashtags with specific tweet (before cleaning)
turkey['Tweet'].iloc[0]


# In[22]:


#cleaning of hashtags

turkey['clean_tweet']=turkey['Tweet'].apply(nfx.remove_hashtags)


# In[23]:


#comparison with old Tweet and cleaned Tweet (HASHTAGS)
turkey[['Tweet','clean_tweet']]


# In[24]:


#after cleaning hashtags display clean tweet
turkey['clean_tweet']


# In[25]:


# first 5 reows 
turkey.head()


# In[26]:


#cleaning extra spaces

#before

turkey['clean_tweet'].iloc[2]


# In[27]:


turkey['clean_tweet'] = turkey['clean_tweet'].apply(nfx.remove_multiple_spaces)


# In[28]:


#after cleaning
turkey['clean_tweet'].iloc[2]


# In[29]:


#cleaning urls
turkey['clean_tweet'] = turkey['clean_tweet'].apply(nfx.remove_urls)


# In[30]:


#after cleaning
turkey['clean_tweet'].iloc[2]


# In[31]:


#cleaning punctuation
turkey['clean_tweet'] = turkey['clean_tweet'].apply(nfx.remove_puncts)


# In[32]:


#cleaning stop words
turkey['clean_tweet'] = turkey['clean_tweet'].apply(nfx.remove_stopwords)


# In[33]:


#remove special character
turkey['clean_tweet'] = turkey['clean_tweet'].apply(nfx.remove_special_characters)


# In[34]:


#final

turkey[['Tweet','clean_tweet']]


# In[35]:


#calculate null values in the data set
turkey.isnull().sum()


# In[36]:


turkey.head(5)


# In[37]:


get_ipython().system('pip install stanfordcorenlp')



# In[39]:


from stanfordcorenlp import StanfordCoreNLP
import json


# In[40]:


# set the CoreNLP path
nlp = StanfordCoreNLP('E:\stanford-corenlp-4.5.2')


# In[40]:


# import multiprocessing


# In[62]:


# import numpy as np


# In[42]:


# import itertools


# In[43]:


# # define a function to get the sentiment of a text

# def get_emotion(text):
#     doc = nlp(text)
#     sentiment = 0
#     for sent in doc.sentences:
#         sentiment += sent.sentiment
#     if len(doc.sentences) > 0:
#         sentiment /= len(doc.sentences)
#     else:
#         return 'unprocessable'
#     if sentiment >= 0.5:
#         return 'joyful'
#     elif sentiment > 0:
#         return 'happy'
#     elif sentiment == 0:
#         return 'neutral'
#     elif sentiment > -0.5:
#         return 'sad'
#     else:
#         return 'angry'

# def get_emotions_parallel(df):
#     # define number of chunks and processes
#     NUM_CHUNKS = 4
#     NUM_PROCESSES = multiprocessing.cpu_count()
    
#     # define function to get emotions in parallel
#     chunks = np.array_split(df, NUM_CHUNKS)
#     with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
#         results = pool.map(get_emotion, itertools.chain.from_iterable([chunk['clean_tweet'].tolist() for chunk in chunks]))
        
#     # return the results
#     return results
    
# turkey['Score1'] = get_emotions_parallel(turkey)


    


# In[63]:


# # define a function to get the sentiment of a text
# def get_emotion(text):
#     doc = nlp(text)
#     sentiment = 0
#     for sent in doc.sentences:
#         sentiment += sent.sentiment
#     if len(doc.sentences) > 0:
#         sentiment /= len(doc.sentences)
#     else:
#         return 'unprocessable'
#     if sentiment >= 0.5:
#         return 'joyful'
#     elif sentiment > 0:
#         return 'happy'
#     elif sentiment == 0:
#         return 'neutral'
#     elif sentiment > -0.5:
#         return 'sad'
#     else:
#         return 'angry'


# In[44]:


def get_emotion(text):
    # define the CoreNLP properties
    props = {
        'annotators': 'sentiment',
        'outputFormat': 'json',
        'timeout': 10000,
    }
    # analyze the sentiment of the text
    output = nlp.annotate(text, properties=props)
    # parse the output
    try:
        sentiment = json.loads(output)['sentences'][0]['sentiment']
    except IndexError:
        return 'unknown'
    if sentiment == 'Positive':
        return 'joyful'
    elif sentiment == 'Neutral':
        return 'happy'
    elif sentiment == 'Negative':
        return 'sad'
    else:
        return 'angry'



# In[45]:


# apply the function to the clean_tweet
turkey['Score1'] = turkey['clean_tweet'].apply(get_emotion)


# In[46]:


turkey.head(5)


# In[47]:


turkey.tail(5)


# In[48]:


#calculating percentage of happy tweets 
happy=turkey[turkey['Score1']=="happy"]
print(str(happy.shape[0]/(turkey.shape[0])*100)+" % Happy Tweets")
Happy=happy.shape[0]/turkey.shape[0]*100


# In[49]:


#calculating percentage of angry tweets
angry=turkey[turkey['Score1']=="angry"]
print(str(angry.shape[0]/(turkey.shape[0])*100)+" % Angry Tweets")
Angry=angry.shape[0]/turkey.shape[0]*100


# In[50]:


#calculating percentage of sad tweets
sad=turkey[turkey['Score1']=="sad"]
print(str(sad.shape[0]/(turkey.shape[0])*100)+" % Sad Tweets")
Sad=sad.shape[0]/turkey.shape[0]*100


# In[51]:


#calculating percentage of joyful tweets
joyful=turkey[turkey['Score1']=="joyful"]
print(str(joyful.shape[0]/(turkey.shape[0])*100)+" % Joyful Tweets")
Joyful=joyful.shape[0]/turkey.shape[0]*100


# In[52]:


#calculating percentage of neutral tweets
neutral=turkey[turkey['Score1']=="neutral"]
print(str(neutral.shape[0]/(turkey.shape[0])*100)+" % Neutral Tweets")
Neutral=neutral.shape[0]/turkey.shape[0]*100


# In[69]:


# Pie chart

labels = ['Angry', 'Happy', 'Sad', 'Joyful']
sizes = [Angry, Happy, Sad, Joyful]
colors = ['#ff9999','#66b3ff','#99ff99','#ffcccc']
explode = (0, 0, 0, 0.1)

fig1, ax1 = plt.subplots(figsize=(9, 9))
ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', 
        shadow=True, startangle=0)
ax1.axis('equal')
# plt.xlabel("Sentiments")
#plt.ylabel("Percentage")
plt.title("Sentiment Analysis on Turkey earthquake", fontweight='bold')
plt.legend(title='Sentiment', labels=labels, loc="upper left", bbox_to_anchor=(1,1), fontsize=12)
plt.show()


# In[61]:


#bar chart

labels = turkey.groupby('Score1').count().index.values #(how many values fall in happy cateogery)
values = turkey.groupby('Score1').size().values #sizes(happy,depressed,neutral)
plt.bar(labels,values)


# In[64]:


get_ipython().system('pip install scikit-learn')


# In[70]:


#performance evaluation
from sklearn.model_selection import train_test_split




# In[71]:


train_data, test_data = train_test_split(turkey, test_size=0.3, random_state=42)


# In[ ]:


test_data['predicted'] = test_data['clean_tweet'].apply(get_emotion)


# In[ ]:


from sklearn.metrics import accuracy_score
score = accuracy_score(test_data['Score1'], test_data['predicted'])
print("Accuracy: ", score)


# In[ ]:




