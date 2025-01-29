import numpy as np
import pandas as pd
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('punkt')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')
import streamlit as st
from PIL import Image

amazon_df = pd.read_csv('amazon_product.csv')
amazon_df.head()

amazon_df.drop('id', axis=1, inplace=True)


stemmer = SnowballStemmer('english')
def tokenize_stem(text):
  tokens = nltk.word_tokenize(text.lower())
  stemmed = [stemmer.stem(w) for w in tokens]
  return " ".join(stemmed)

amazon_df['stemmed_tokens'] = amazon_df.apply(
    lambda row: tokenize_stem(row['Title'] + ' ' + row['Description']),
    axis=1
)

tfidfv = TfidfVectorizer(tokenizer=tokenize_stem)

def cosine_sim(txt1, txt2):
  matrix = tfidfv.fit_transform([txt1, txt2])
  return cosine_similarity(matrix)[0][1]

def search_product(query):
  stemmed_query = tokenize_stem(query)
  #calculate cosine similarity bw query and stemmed tokens
  amazon_df['similarity'] = amazon_df['stemmed_tokens'].apply(lambda x: cosine_sim(stemmed_query, x))
  res = amazon_df.sort_values(by=['similarity'], ascending=True).head(10)[['Title', 'Description', 'Category']]
  return res

img = Image.open("img.jpg")
st.image(img, width=600)
st.title("Search Engine and Product Recommendation System")

query = st.text_input("Enter your product")
submit = st.button('Search')
if submit:
  res = search_product(query)
  st.write(res)
