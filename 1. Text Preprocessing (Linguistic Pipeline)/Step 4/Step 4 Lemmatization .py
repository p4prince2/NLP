#!/usr/bin/env python
# coding: utf-8

# # ***Step 4: Lemmatization***
# 
# Lemmatization is quite similar to the Stamming. It is used to group different inflected forms of the word, called Lemma. The main difference between Stemming and lemmatization is that it produces the root word, which has a meaning.
# 
# ***For example:*** In lemmatization, the words intelligence, intelligent, and intelligently has a root word intelligent, which has a meaning.

# # **Lemmatization in NLP**
# Lemmatization is the process of reducing words to their **base or dictionary form** (lemma) while ensuring the result is a valid word.
# Unlike stemming, which simply chops off suffixes, lemmatization considers **context and grammar** to produce a meaningful word.
# 
# ### **🔹 Difference Between Lemmatization and Stemming**
# | Feature | Stemming | Lemmatization |
# |---------|---------|--------------|
# | Method | Removes suffixes using rules | Uses a dictionary to get the root form |
# | Output | Can produce non-meaningful words | Always returns a valid word |
# | Example ('running') | `runn` (not a valid word) | `run` (correct) |
# | Example ('better') | `better` (unchanged) | `good` (meaning-based) |
# | Example ('wolves') | `wolv` (incorrect) | `wolf` (correct) |
# | Speed | Faster (rule-based) | Slower (dictionary-based) |
# 

# In[25]:


import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required resources
nltk.download('punkt')
nltk.download('wordnet')

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()


# ### **🔹 Example of Lemmatization in Python**

# In[10]:


# Example words
words = ['running', 'flies', 'better', 'children', 'wolves', 'studies']

# Apply Lemmatization
lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

print('Original Words:', words)
print('Lemmatized Words:', lemmatized_words)


# ### **🔹 Expected Output**
# ```
# Original Words: ['running', 'flies', 'better', 'children', 'wolves', 'studies']
# Lemmatized Words: ['running', 'fly', 'better', 'child', 'wolf', 'study']
# ```
# ✅ Notice that 'better' remains unchanged because **lemmatization needs POS tagging** to recognize that its lemma is 'good'.
# 

# # ***Improving Lemmatization with POS Tagging***
# 
#  -Lemmatization works better when you specify the Part of Speech (POS):
# 
#  -Noun (n) → "children" → "child"
# 
#  -Verb (v) → "running" → "run"
# 
#  -Adjective (a) → "better" → "good"

# In[43]:


word=WordNetLemmatizer()

word.lemmatize("going",pos='v')


# ### **🔹 When to Use Lemmatization?**
# - ✅ When you need **meaningful** words, not just root forms.
# - ✅ For **Natural Language Processing (NLP)** tasks like **chatbots, text analysis, and AI**.
# - ✅ When working with **search engines** (lemmatization improves search accuracy).
# 

# ### **🔹 Conclusion**
# - **Stemming is fast** but may produce incorrect words (e.g., 'running' → 'runn').
# - **Lemmatization is slower** but **more accurate** and returns **real words**.
# - **Using POS tagging** improves lemmatization significantly.
# 

# In[ ]:





# ***

# ***

# 
# ## ***Stemming vs Lemmatization***
# # This notebook explains the difference between **Stemming** and **Lemmatization** in Natural Language Processing (NLP)."

# ## 1️⃣ Stemming
# 
# ### Definition:
# - **Stemming** is a **rule-based process** that reduces words to their **root form** by simply removing prefixes and suffixes. 
# - It often leads to **non-dictionary** words, which may not be valid.
# 
# ### Approach:
# - Stemming uses **heuristic rules** (such as removing common suffixes like "ing", "ed", "es") to chop off word endings.
# - The result may not be an actual word, but a **stem**.
# 
# ### Example:
# - "Running" → "Run"
# - "Happiness" → "Happi"
# - "Loves" → "Lov"
# 
# ### Advantages:
# - **Faster** than lemmatization.
# - Suitable for **simple text analysis** tasks where **accuracy** isn't crucial.
# 
# ### Disadvantages:
# - Can produce **nonsense words** (not real words).
# - Doesn't consider **context** or **meaning**.
# 
# ### Tools:
# - **Porter Stemmer** (most common)
# - **Lancaster Stemmer** (aggressive)
# - **Snowball Stemmer** (improved version of Porter)
# 

# ## 2️⃣ Lemmatization
# 
# ### Definition:
# - **Lemmatization** is a **dictionary-based** process that converts words into their **base or root form** (known as the "lemma").
# - It uses a **morphological analysis** to consider the **context** and **part of speech (POS)**.
# 
# ### Approach:
# - Lemmatization looks at the word and its context to find the **correct lemma**. This ensures that the lemmatized word is a **valid word** in the dictionary.
# 
# ### Example:
# - "Running" → "Run"
# - "Happiness" → "Happiness"
# - "Loves" → "Love"
# 
# ### Advantages:
# - Produces **real words** (dictionary forms).
# - **Context-aware**: Takes into account the part of speech to provide correct lemmatization.
# - **Accurate** and **more sophisticated** than stemming.
# 
# ### Disadvantages:
# - **Slower** than stemming.
# - Requires more **computational resources** because it uses a **lexical database** (e.g., **WordNet**).
# 
# ### Tools:
# - **WordNet Lemmatizer** (NLTK)
# - **spaCy Lemmatizer** (more advanced, efficient)
# 

# ## 🔹 Comparison Table: Stemming vs Lemmatization
# 
# | Feature           | **Stemming**                         | **Lemmatization**                         |
# |-------------------|--------------------------------------|-------------------------------------------|
# | **Method**        | Rule-based (removes suffixes)        | Dictionary-based (morphological analysis) |
# | **Speed**         | Faster                               | Slower                                    |
# | **Accuracy**      | Less accurate (may produce invalid words) | More accurate (valid words)               |
# | **Result**        | May produce **non-dictionary** words | Always produces **real dictionary words** |
# | **Use Case**      | When speed is prioritized (e.g., search engines, quick text preprocessing) | When **meaning** and **accuracy** matter (e.g., sentiment analysis, NLP tasks) |
# | **Example**       | "Running" → "Run" (incorrect for verbs) | "Running" → "Run" (correct)               |
# | **Tools**         | Porter, Snowball, Lancaster Stemmer  | WordNet Lemmatizer, spaCy, TextBlob       |
# 

# In[ ]:




