#!/usr/bin/env python
# coding: utf-8

# # ***Step3: Stemming***
# 
# Stemming is used to normalize words into its base form or root form. For example, celebrates, celebrated and celebrating, all these words are originated with a single root word "**celebrate**." The big problem with stemming is that sometimes it produces the root word which may not have any meaning.
# 
# ***For Example***, intelligence, intelligent, and intelligently, all these words are originated with a single root word "**intelligen**." In English, the word "intelligen" do not have any meaning.

# 
# ## Stemming in Text Preprocessing
# 
# ### What is Stemming?
# Stemming is a text preprocessing technique used in Natural Language Processing (NLP) 
# to reduce words to their root form by removing suffixes. It helps normalize words, 
# making them easier to process and analyze in machine learning models, search engines, 
# and text-based applications.
# 
# Example:
# - "running" → "run"
# - "flies" → "fli" (instead of "fly")
# - "better" → "better" (does not change because "better" is not modified correctly by traditional stemming)
# 
# ### Why Use Stemming?
# - Reduces vocabulary size by grouping similar words together.
# - Improves model efficiency by reducing redundant variations of words.
# - Helps in text classification, sentiment analysis, and search engines by making words comparable.
# 
# 
# ### Types of Stemmers
# 1. Porter Stemmer (Most common, rule-based)
# 2. Lancaster Stemmer (More aggressive)
# 3. Snowball Stemmer (Improved version of Porter Stemmer)
# 
# 
# ### Limitations of Stemming
# - It is crude and may not always produce actual words.
# - Words like "flies" become "fli" instead of "fly."
# - "Better" remains unchanged due to limitations in rule-based approaches.
# - Not as accurate as **lemmatization**, which uses dictionary-based word forms.
# 
# 
# 
# ### When to Use Stemming?
# - When speed is more important than linguistic accuracy.
# - In applications like search engines where approximate matching is sufficient.
# - When working with large datasets and computational efficiency is a concern.

# In[60]:


from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer

def porter_stemmer_example(text):
    """
    Stem words using the Porter Stemmer.
    
    The Porter Stemmer is a widely used rule-based stemming algorithm that removes suffixes 
    from words to reduce them to their root form. It is considered to be effective for English text.
    
    Args:
    text (str): The input text to be stemmed.
    
    Returns:
    list: A list of words after stemming using the Porter Stemmer.
    """
    # Create an instance of the PorterStemmer
    porter = PorterStemmer()
    
    # Split the text into words and stem each word
    words = text.split()
    stemmed_words = [porter.stem(word) for word in words]
    
    return stemmed_words


def lancaster_stemmer_example(text):
    """
    Stem words using the Lancaster Stemmer.
    
    The Lancaster Stemmer is a more aggressive stemmer compared to the Porter Stemmer. 
    It uses a different set of rules and might reduce words to their root form faster and 
    sometimes to a greater degree than the Porter Stemmer.
    
    Args:
    text (str): The input text to be stemmed.
    
    Returns:
    list: A list of words after stemming using the Lancaster Stemmer.
    """
    # Create an instance of the LancasterStemmer
    lancaster = LancasterStemmer()
    
    # Split the text into words and stem each word
    words = text.split()
    stemmed_words = [lancaster.stem(word) for word in words]
    
    return stemmed_words


def snowball_stemmer_example(text, language='english'):
    """
    Stem words using the Snowball Stemmer.
    
    The Snowball Stemmer is an improved version of the Porter Stemmer with better 
    handling of irregular words. It also supports multiple languages, making it more versatile.
    
    Args:
    text (str): The input text to be stemmed.
    language (str): The language of the text (default is 'english').
    
    Returns:
    list: A list of words after stemming using the Snowball Stemmer.
    """
    # Create an instance of the SnowballStemmer
    snowball = SnowballStemmer(language)
    
    # Split the text into words and stem each word
    words = text.split()
    stemmed_words = [snowball.stem(word) for word in words]
    
    return stemmed_words


# Example Usage
if __name__ == "__main__":
    text = "running runner runs easily cared cars and fairly, sportingly"
    
    print("Porter Stemmer:", porter_stemmer_example(text))
    print("Lancaster Stemmer:", lancaster_stemmer_example(text))
    print("Snowball Stemmer:", snowball_stemmer_example(text))


# # **RegexStemmer in NLTK**
# This notebook demonstrates how to use the `RegexpStemmer` from NLTK for stemming words based on custom regular expressions.
# 
# **Key Features of RegexpStemmer:**
# - Allows defining custom regex patterns for stemming.
# - Provides better control over which suffixes to remove.
# - Faster than traditional stemming methods like Porter and Snowball.

# ## **Define and Use RegexpStemmer**
# We will create a `RegexpStemmer` that removes common suffixes such as `-ing`, `-ed`, and `-ly`. The `min=4` parameter ensures that we do not over-stem words, meaning the root should have at least 4 characters left after stemming.
# 
# # Define a RegexpStemmer with different regex patterns
# 
# ***1️⃣ Removing only suffixes (at the end of the word)***
# 
#  - The `$` symbol ensures that only the ending matches are removed.
#  - Example: "running" → "runn", "jumped" → "jump", "edited" → "edit"
# 
# ***2️⃣ Removing only prefixes (at the start of the word)***
# 
#  - The `^` symbol ensures that only the beginning matches are removed.
#  - Example: "unhappy" → "happy", "unable" → "able"
# 
# ***3️⃣ Removing a pattern anywhere in the word (no `^` or `$`)***
#     
#  - Removes "ed" wherever it appears in the word.
#  - Example: "edited" → "it", "education" → "ucation", "bed" → "b"

# In[23]:


from nltk.stem import RegexpStemmer


# In[25]:


def regex_stemmer_example(words):
    """
    Apply regex-based stemming to a list of words using RegexpStemmer.

    The regex pattern removes '-ing', '-ed', and '-ly' suffixes only if the root
    word has at least 4 characters left.

    Args:
    words (list): A list of words to be stemmed.

    Returns:
    list: The list of stemmed words.
    """
    # Define the regex-based stemmer
    regex_stemmer = RegexpStemmer(r'ing$|ed$|ly$', min=4)

    # Apply stemming to each word
    stemmed_words = [regex_stemmer.stem(word) for word in words]

    return stemmed_words


# ## **Example Usage**
# Let's apply our `RegexpStemmer` to a list of sample words and observe the transformations.

# In[28]:


# Example words
words = ['running', 'jumped', 'happily', 'caring', 'studied', 'flies']


# Apply Regex Stemmer
stemmed_words = regex_stemmer_example(words)

# Display results
print('Original Words:', words)
print('Stemmed Words:', stemmed_words)


# # **Or in simple**

# In[40]:


## or in simple
reg_stemm=RegexpStemmer('ing$|s$|e$|able$', min=4)
print(reg_stemm.stem("eating"))
print(reg_stemm.stem("cars"))
print(reg_stemm.stem("excessive"))


# ## **Conclusion**
# - `RegexpStemmer` provides a customizable way to stem words using regex.
# - The `min` parameter prevents excessive reduction of word stems.
# - This method is ideal for domain-specific text processing where predefined stemming rules are needed.
# 
# - $ at the end → Removes suffix only.
# 
# - ^ at the start → Removes prefix only.
# 
# - No ^ or $ → Removes the pattern anywhere in the word.

# In[ ]:





# In[ ]:




