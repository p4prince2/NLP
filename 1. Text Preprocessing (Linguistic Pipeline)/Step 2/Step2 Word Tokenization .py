#!/usr/bin/env python
# coding: utf-8

# # ***Word Tokenizer is used to break the sentence into separate words or tokens.***
# 
# ***Example:***
# 
# JavaTpoint offers Corporate Training, Summer Training, Online Training, and Winter Training.
# 
# ***Word Tokenizer generates the following result:***
# 
# "JavaTpoint", "offers", "Corporate", "Training", "Summer", "Training", "Online", "Training", "and", "Winter", "Training", "."

# In[19]:


##  paragraph ---> Words
## Sentence --> words

from nltk.tokenize import word_tokenize


# In[13]:


paragraph ='''Independence Day is one of the important festivals for every Indian citizen. It is celebrated on the 15th of August each year ever since India got independence
from the British rule.\nThe day celebrates independence in the true sense.'''


# In[21]:


sentence="JavaTpoint offers Corporate Training, Summer Training, Online Training, and Winter Training."


# In[15]:


paragraph


# In[17]:


word_tokenize(paragraph)


# In[23]:


word_tokenize(sentence)


# In[25]:


## word_tokenize(), it splits words at punctuation and treats punctuation as separate tokens.

from nltk.tokenize import wordpunct_tokenize


# In[27]:


print(wordpunct_tokenize(paragraph))
print(wordpunct_tokenize(sentence))


# In[29]:


from nltk.tokenize import TreebankWordTokenizer


# ![image.png](attachment:7ca5ad09-bdb1-4adb-89db-2a3404e220a1.png)

# In[31]:


token=TreebankWordTokenizer()


# In[33]:


token.tokenize(paragraph)


# In[ ]:




