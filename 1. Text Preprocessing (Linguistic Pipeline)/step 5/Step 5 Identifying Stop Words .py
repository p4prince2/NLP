#!/usr/bin/env python
# coding: utf-8

# ## ***Step 5: Identifying Stop Words***
# 
# In English, there are a lot of words that appear very frequently like "is", "and", "the", and "a". NLP pipelines will flag these words as stop words. Stop words might be filtered out before doing any statistical analysis.
# 
# ***Example:*** He is a good boy.
# 
# **Note:** When you are building a rock band search engine, then you do not ignore the word "The."

# ## Identifying Stop Words in NLP
# 
# Stop words are commonly used words (such as “the,” “is,” “in,” and “and”) that carry little meaningful information in the context of text analysis. Removing these stop words can help improve the accuracy and performance of many NLP tasks, such as text classification and sentiment analysis.

# ## Why Identify Stop Words?
# 
# Stop words are typically removed from text data to:
# 
# * Improve model performance by focusing on the more meaningful words.
# * Reduce computational complexity because processing fewer words is faster.
# * Enhance the quality of features used in machine learning models, as stop words tend to dominate the dataset without contributing valuable information.

# ## How to Identify Stop Words?
# 
# In Python, the NLTK (Natural Language Toolkit) library provides a built-in list of stop words for various languages. Here's how to identify and remove stop words:

# ## Steps to Identify and Remove Stop Words Using NLTK
# 
# 1.  **Install and Import NLTK Library**
# 
#     First, you'll need to install the NLTK library (if you haven't already):
# 
#     ```bash
#     pip install nltk
#     ```
# 
#     Then, import the necessary modules:
# 
#     ```python
#     import nltk
#     from nltk.corpus import stopwords
#     
#     nltk.download('stopwords')
# nltk.download('punkt')
#     ```
# 
# 2.  **Accessing Stop Words in NLTK**
# 
#     NLTK provides a list of stop words for various languages. Here's how you can access and view the stop words in English:
# 
#     ```python
#     stop_words = set(stopwords.words('english'))
# 
#     # Print a sample of stop words
#     print(list(stop_words)[:10])
#     ```
# 
#     This will output the first 10 stop words in English, such as:
# 
#     ```
#     ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your']
#     ```

# ## 3. Removing Stop Words from Text
# 
# Once you've identified the stop words, you can remove them from your text data:
# 
# ```python
# # Sample sentence
# sentence = "This is a sample sentence to demonstrate stop word removal."
# 
# # Tokenize the sentence into words
# words = sentence.split()
# 
# # Filter out stop words
# filtered_words = [word for word in words if word.lower() not in stop_words]
# 
# print("Original Sentence:", sentence)
# print("Filtered Sentence:", " ".join(filtered_words))
# ```
# 
# Output:
# 
# ```
# Original Sentence: This is a sample sentence to demonstrate stop word removal.
# Filtered Sentence: sample sentence demonstrate stop word removal.
# ```
# 
# ## Custom Stop Words List
# 
# Sometimes, you may want to add custom stop words or remove certain words from the default list provided by NLTK. Here's how you can do that:
# 
# ```python
# # Add custom stop words
# custom_stopwords = stop_words.union({'demonstrate', 'sample'}) # Assuming 'stop_words' is already defined
# 
# # Filter text using custom stopwords
# filtered_custom = [word for word in words if word.lower() not in custom_stopwords]
# 
# print("Filtered with custom Stopwords:", " ".join(filtered_custom))
# ```

# ## Common Stop Words (examples)
# 
# * Articles: "a", "an", "the"
# * Prepositions: "in", "on", "at", "by", "with"
# * Pronouns: "I", "you", "he", "she", "it", "they"
# * Conjunctions: "and", "but", "or", "nor", "for"
# * Auxiliary Verbs: "is", "am", "are", "was", "were"
# 
# ## Use Cases for Stop Word Removal
# 
# * **Text Classification**: Removing stop words improves the classification process as the algorithm will focus on more meaningful words.
# * **Sentiment Analysis**: Stop word removal helps in identifying the sentiment of a text by focusing on words that carry sentiment (positive or negative).
# * **Search Engines**: Stop words are ignored in search queries to improve search speed and relevance.
# 
# ## Conclusion
# * Stop word removal is a common preprocessing step in NLP that helps in reducing noise and improving the quality of text data for analysis.
# * NLTK offers a simple way to handle stop words, and you can also customize your stop word list based on specific requirements.

# 
# 
# ## Example:
# 

# In[8]:


stopwords.words("english") ## list of stop words in english


# In[12]:


stopwords.words("french") ## list of stop words in FRENCH


# In[ ]:




