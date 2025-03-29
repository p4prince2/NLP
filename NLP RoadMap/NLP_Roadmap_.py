#!/usr/bin/env python
# coding: utf-8

# # Roadmap to Learn NLP
# 
# This notebook provides a structured roadmap for learning Natural Language Processing (NLP), 
# starting from basic text processing techniques to advanced deep learning models like Transformers (BERT, GPT).
# 
# We will also visualize the roadmap as a **pyramid structure** using Python's Matplotlib and NetworkX.
# 

# 
# ## 1. Cleaning Input (Preprocessing)
# - **Techniques**: Tokenization, Lemmatization, Stemming
# - **Purpose**: Converts raw text into a cleaner, more usable format.
# 
# ## 2. Converting Input Text to Vector Representations
# - **BoW (Bag of Words)**: Counts word occurrences.
# - **TF-IDF (Term Frequency-Inverse Document Frequency)**: Weighs words based on importance.
# - **Unigram Models**: Uses single-word sequences.
# 
# ## 3. Advanced Text Vectorization
# - **Word2Vec**: Converts words into dense vectors based on context.
# - **GloVe**: Captures word meaning based on co-occurrence.
# - **FastText**: Similar to Word2Vec but considers subword information.
# 
# ## 4. Recurrent Neural Networks (RNNs)
# - **LSTM (Long Short-Term Memory)**: Handles long-term dependencies.
# - **GRU (Gated Recurrent Units)**: A simplified LSTM with fewer parameters.
# 
# ## 5. Word Embeddings & Contextual Representations
# - **Pre-trained models like Word2Vec, GloVe, FastText**.
# 
# ## 6. Transformer Models
# - **Attention-based models replacing RNNs**.
# - **More efficient for parallel computation**.
# 
# ## 7. Advanced Deep Learning Models for NLP
# - **BERT (Bidirectional Encoder Representations from Transformers)**
# - **GPT (Generative Pre-trained Transformer)**
# 

# In[1]:


import matplotlib.pyplot as plt
import networkx as nx

# Define pyramid structure with explicit hierarchy levels
stages = [
    ("Cleaning Input", "Input Text to Vector (BoW, TF-IDF, Unigram)"),
    ("Input Text to Vector (BoW, TF-IDF, Unigram)", "Input Text to Vector (Word2Vec, GloVe)"),
    ("Input Text to Vector (Word2Vec, GloVe)", "RNN, LSTM, GRU"),
    ("RNN, LSTM, GRU", "Word Embeddings"),
    ("Word Embeddings", "Transformer"),
    ("Transformer", "BERT, GPT")
]

# Create graph
G = nx.DiGraph()
G.add_edges_from(stages)

# Define node levels for proper layout
node_levels = {
    "Cleaning Input": 0,
    "Input Text to Vector (BoW, TF-IDF, Unigram)": 1,
    "Input Text to Vector (Word2Vec, GloVe)": 2,
    "RNN, LSTM, GRU": 3,
    "Word Embeddings": 4,
    "Transformer": 5,
    "BERT, GPT": 6
}

plt.figure(figsize=(10, 6))
pos = {node: (0, level) for node, level in node_levels.items()}  # Vertical pyramid layout

nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=3000, font_size=10, font_weight='bold')
plt.title("NLP Learning Roadmap", fontsize=14)
plt.show()


# In[ ]:




