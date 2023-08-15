#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os


# In[9]:


import nltk


# In[12]:


#nltk.download()


# In[13]:


AI = '''Artificial Intelligence refers to the intelligence of machines. This is in contrast to the natural intelligence of 
humans and animals. With Artificial Intelligence, machines perform functions such as learning, planning, reasoning and 
problem-solving. Most noteworthy, Artificial Intelligence is the simulation of human intelligence by machines. 
It is probably the fastest-growing development in the World of technology and innovation. Furthermore, many experts believe
AI could solve major challenges and crisis situations.'''


# In[14]:


AI


# In[15]:


type(AI)


# In[16]:


from nltk.tokenize import word_tokenize


# In[17]:


AI_tokens=word_tokenize(AI)


# In[18]:


AI_tokens


# In[19]:


len(AI_tokens)


# In[20]:


from nltk.tokenize import sent_tokenize


# In[21]:


AI_sent=sent_tokenize(AI)


# In[22]:


AI_sent


# In[23]:


len(AI_sent)


# In[24]:


AI


# In[25]:


from nltk.tokenize import blankline_tokenize


# In[26]:


AI_blank=blankline_tokenize(AI)


# In[27]:


AI_blank


# In[28]:


len(AI_blank)


# In[29]:


from nltk.util import bigrams,trigrams,ngrams


# In[30]:


string='the best and most beautiful thing in the world cannot be seen or even touched,they must be felt with heart'


# In[31]:


quotes_tokens=nltk.word_tokenize(string)


# In[32]:


quotes_tokens


# In[33]:


len(quotes_tokens)


# In[34]:


quotes_bigrams=list(nltk.bigrams(quotes_tokens))


# In[35]:


quotes_bigrams


# In[36]:


quotes_tokens


# In[37]:


quotes_trigrams=list(nltk.trigrams(quotes_tokens))


# In[38]:


quotes_trigrams


# In[39]:


quotes_ngrams=list(nltk.ngrams(quotes_tokens))


# In[40]:


quotes_ngrams=list(nltk.ngrams(quotes_tokens,4))


# In[41]:


quotes_ngrams


# In[42]:


len(quotes_tokens)


# In[43]:


quotes_ngrams_1=list(nltk.ngrams(quotes_tokens,5))


# In[44]:


quotes_ngrams_1


# In[45]:


#porter stemmer


# In[46]:


from nltk.stem import PorterStemmer


# In[47]:


pst=PorterStemmer()


# In[48]:


pst


# In[49]:


pst
pst.stem('having')


# In[50]:


pst.stem('affection')


# In[51]:


pst.stem('playing')


# In[52]:


pst.stem('give')


# In[53]:


words_to_stem=['give','giving','given','gave']
for words in words_to_stem:
    print(words+':'+pst.stem(words))


# In[54]:


words_to_stem=['give','giving','given','gave','thinking','loving','final','finalized','finally']


# In[55]:


for words in words_to_stem:
    print(words+':'+pst.stem(words))


# In[56]:


from nltk.stem import LancasterStemmer
lst = LancasterStemmer()
for words in words_to_stem:
    print(words + ':' + lst.stem(words))


# In[57]:


from nltk.stem import SnowballStemmer


# In[58]:


sbst=SnowballStemmer('english')


# In[59]:


sbst


# In[60]:


for words in words_to_stem:
    print(words+':'+sbst.stem(words))


# In[61]:


from nltk.stem import wordnet


# In[62]:


from nltk.stem import WordNetLemmatizer


# In[63]:


word_lem=WordNetLemmatizer()


# In[64]:


words_to_stem


# In[66]:


for words in words_to_stem:
    print(words+':'+word_lem.lemmatize(words))


# In[67]:


pst.stem('final')


# In[68]:


lst.stem('finally')


# In[69]:


sbst.stem('finalized')


# In[70]:


lst.stem('final')


# In[71]:


lst.stem('finalized')


# In[ ]:


#Stop words


# In[73]:


from nltk.corpus import stopwords


# In[74]:


stopwords.words('english')


# In[75]:


len(stopwords.words('english'))


# In[76]:


stopwords.words('spanish')


# In[77]:


len(stopwords.words('spanish'))


# In[78]:


stopwords.words('french')


# In[79]:


len(stopwords.words('french'))


# In[80]:


stopwords.words('german')


# In[81]:


len(stopwords.words('german'))


# In[82]:


stopwords.words('hindi')


# In[83]:


import re


# In[84]:


punctuation=re.compile(r'[-.?,:;()|0-9]')


# In[85]:


punctuation


# In[86]:


AI


# In[87]:


AI_tokens


# In[88]:


len(AI_tokens)


# In[ ]:


#POS using NLTK library


# In[89]:


sent='kathy is a natural when it comes to drawing'


# In[90]:


sent


# In[91]:


sent_tokens=word_tokenize(sent)


# In[92]:


sent_tokens


# In[98]:


for token in sent_tokens:
    print(nltk.pos_tag([token]))


# In[101]:


sent2='john is eating a delicious cake'
sent2_tokens=word_tokenize(sent2)


# In[102]:


for token in sent2_tokens:
    print(nltk.pos_tag([token]))


# In[103]:


from nltk import ne_chunk


# In[104]:


NE_sent='The US president stays in the WHITEHOUSE'


# In[105]:


NE_sent


# In[107]:


NE_tokens=word_tokenize(NE_sent)


# In[108]:


NE_tokens


# In[109]:


NE_tags=nltk.pos_tag(NE_tokens)


# In[110]:


NE_tags


# In[111]:


NE_NER=ne_chunk(NE_tags)


# In[112]:


print(NE_NER)


# In[113]:


new='the big cat ate the little mouse who was after fresh cheese'


# In[114]:


new


# In[115]:


new_tokens=nltk.pos_tag(word_tokenize(new))


# In[116]:


new_tokens


# In[ ]:


#Libraries


# In[140]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt


# In[133]:


import matplotlib.pyplot as plt


# In[141]:


text=('Python Python Python Matplotlib Matplotlib Seaborn Network Plot Violin Chart Pandas Datassience Wordcloud Spider Radar Parrallel Alpha Color Brewer Density Scatter Barplot Barplot Boxplot Violinplot Treemap Stacked Area Chart Chart Visualization Dataviz Donut Pie Time-Series Wordcloud Wordcloud Sankey Bubble")')


# In[142]:


text


# In[ ]:


#create the wordcloud object


# In[143]:


wordcloud=WordCloud(width=480,height=480,margin=0).generate(text)


# In[144]:


plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()


# In[ ]:





# In[ ]:




