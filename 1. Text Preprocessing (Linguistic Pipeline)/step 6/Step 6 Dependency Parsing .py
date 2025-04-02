#!/usr/bin/env python
# coding: utf-8

# ## Step 6: Dependency Parsing
# 
# Dependency Parsing is used to find that how all the words in the sentence are related to each other.

# # Dependency Parsing in NLP
# 
# **Dependency parsing** is a natural language processing (NLP) technique used to analyze the grammatical structure of a sentence by establishing relationships between words. It helps in understanding how words are connected, which is useful for tasks like **information extraction, question answering, and machine translation**.
# 
# ## **Key Concepts in Dependency Parsing**
# 
# - **Head and Dependent** – Every word in a sentence (except the root) depends on another word called its **head**.
# - **Dependency Relations** – The type of relationship between the head and its dependent (e.g., subject, object, modifier).
# - **Root** – The main verb of the sentence, which serves as the central node in the dependency tree.
# 
# ## **Example**
# 
# ### **Sentence:**  
# **"The cat sat on the mat."**  
# 
# ### **Dependency Parse:**  
# - **sat** (ROOT)  
# - **cat** → subject of "sat" (**nsubj**)  
# - **on** → preposition modifying "sat" (**prep**)  
#   - **mat** → object of "on" (**pobj**)  
# - **The** → determiner modifying "cat" (**det**)  
# - **The** → determiner modifying "mat" (**det**)  
# 
# ## **Implementation in Python (Using spaCy)**
# 

# In[2]:


get_ipython().system('pip install spacy')


# In[35]:


import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Sample sentence
sentence = "The cat sat on the mat."

# Process the sentence
doc = nlp(sentence)

# Print dependency relations
for token in doc:
    print(f"{token.text} --> {token.dep_} --> {token.head.text}")


# ### **Expected Output:**  
# ```
# The --> det --> cat
# cat --> nsubj --> sat
# sat --> ROOT --> sat
# on --> prep --> sat
# the --> det --> mat
# mat --> pobj --> on
# . --> punct --> sat
# ```
# This output shows how each word in the sentence is related to another in the dependency tree.
# 

# # spaCy vs. NLTK - A Comparison
# 
# Both **spaCy** and **NLTK** are popular **Natural Language Processing (NLP) libraries**, but they serve different purposes. This notebook provides a detailed comparison.
# 
# ## **1. Overview**
# | Feature  | spaCy  | NLTK  |
# |----------|--------|-------|
# | **Purpose** | Industrial-grade NLP | Research & educational NLP |
# | **Speed** | Faster (optimized in Cython) | Slower (pure Python implementation) |
# | **Ease of Use** | Simple, efficient API | More modular but complex |
# | **Pre-trained Models** | Yes (e.g., `en_core_web_sm`) | No (requires external models) |
# | **Deep Learning Integration** | Supports deep learning (via `spacy-transformers`) | Limited support |
# | **Use Case** | Production-ready applications | Academic research & prototyping |
# 
# ## **2. Feature Comparison**
# | Feature | spaCy | NLTK |
# |---------|--------|------|
# | **Tokenization** | Rule-based, fast, supports custom models | Rule-based, flexible but slower |
# | **POS Tagging** | Uses statistical models (accurate) | Uses rule-based and statistical models |
# | **Dependency Parsing** | Yes (built-in, efficient) | Requires Stanford Parser (external) |
# | **Named Entity Recognition (NER)** | Yes (pre-trained models available) | No built-in NER (requires external models) |
# | **Lemmatization** | Fast, model-based | WordNet-based (slower) |
# | **Stemming** | Not available (uses lemmatization instead) | Yes (Porter, Snowball, Lancaster) |
# | **Word Embeddings** | Supports `Word2Vec`, `GloVe`, `fastText`, `transformers` | No direct support |
# | **Sentiment Analysis** | Requires external models | Uses `VADER`, `TextBlob` |
# | **Text Classification** | Supports custom models via `spacy.pipeline` | Uses `nltk.classify` (manual implementation) |
# | **Stopwords** | Inbuilt stopword list | Has stopword lists for multiple languages |
# | **Multi-language Support** | Yes, supports multiple languages | Yes, but relies on external corpora |
# 
# ## **3. Performance & Speed**
# - **spaCy** is significantly **faster** than NLTK.
# - **NLTK** is slower due to its modular nature.
# 
# ## **4. Code Comparison**
# ### **Tokenization**
# 

# ***spaCy***

# In[13]:


import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The quick brown fox jumps over the lazy dog.")

tokens = [token.text for token in doc]
print(tokens)


# 
# ***NLTK***

# In[18]:


import nltk
from nltk.tokenize import word_tokenize

sentence = "The quick brown fox jumps over the lazy dog."
tokens = word_tokenize(sentence)
print(tokens)


# ## **5. When to Use What?**
# | Use Case | Use **spaCy** | Use **NLTK** |
# |----------|--------------|--------------|
# | **Fast NLP processing (production apps)** | ✅ | ❌ |
# | **Machine learning & deep learning integration** | ✅ | ❌ |
# | **Tokenization, POS tagging, dependency parsing** | ✅ | ✅ (but slower) |
# | **Named Entity Recognition (NER)** | ✅ | ❌ |
# | **Educational, research, academic work** | ❌ | ✅ |
# | **Corpus-based NLP experiments** | ❌ | ✅ |
# | **Text classification with custom models** | ✅ | ✅ |
# 
# ## **6. Conclusion**
# - **Use spaCy if you need** a **fast, production-ready, and efficient NLP library**.
# - **Use NLTK if you need** more **customization, linguistic resources, and corpus-based experiments**.
# 
# ### **💡 Recommendation**
# - If you are building **real-world NLP applications** → **Use spaCy** 🚀  
# - If you are doing **NLP research and learning NLP fundamentals** → **Use NLTK** 📚  
# 

# In[ ]:




