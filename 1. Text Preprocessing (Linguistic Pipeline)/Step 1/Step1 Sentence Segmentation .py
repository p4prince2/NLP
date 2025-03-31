#!/usr/bin/env python
# coding: utf-8

# 
# # ***Sentence Segment is the first step for building the NLP pipeline. It breaks the paragraph into separate sentences.***
# 
# 
# ***Example:*** Consider the following paragraph -
# 
# Independence Day is one of the important festivals for every Indian citizen. It is celebrated on the 15th of August each year ever since India got independence from the British rule. The day celebrates independence in the true sense.
# 
# ***Sentence Segment produces the following result:***
# 
# "Independence Day is one of the important festivals for every Indian citizen."
# 
# "It is celebrated on the 15th of August each year ever since India got independence from the British rule."
# 
# "This day celebrates independence in the true sense."

# In[2]:


get_ipython().system('pip install nltk')


# In[1]:


corpus ="""Independence Day is one of the important festivals for every Indian citizen. It is celebrated on the 15th of August each year ever since India got independence from the British rule.
The day celebrates independence in the true sense."""


# In[3]:


corpus


# In[5]:


print(corpus)


# In[17]:


##  paragraph into sentences


from nltk.tokenize import sent_tokenize


# In[19]:


## some time you need to download it 
import nltk
nltk.download('punkt_tab')


# In[21]:


sent_tokenize(corpus)


# In[ ]:




