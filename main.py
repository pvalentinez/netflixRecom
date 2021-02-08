import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

## import data for feature_input
df = pd.read_csv("movie_to_movie.csv")
new_df = df[['title', 'director', 'cast', 'listed_in', 'description']]
new_df.dropna(inplace=True)
blanks = []

## import data for calculate
bag_df = pd.read_csv("bag_of_word.csv")
bag_df.set_index('title', inplace=True)
recommended = []

# CountVector
vectorizer = CountVectorizer()
vectorizer = vectorizer.fit_transform(bag_df['bag_of_words'])
indices = pd.Series(bag_df.index)
cosine = cosine_similarity(vectorizer, vectorizer)

def recommendations(Title, cosine_sim = cosine):
    recommended = []
    idx = indices[indices==Title].index[0]
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=False)
    top_10_indices = list(score_series.iloc[1:11].index)
    
    for i in top_10_indices:
        recommended.append(list(bag_df.index)[i])
    return st.write(recommended)

# Markdown
'''
# Netflix Recommendation
Netflix Recommendation โดยสามารถเลือก **feature** ที่จะนำมาเข้า model ได้
ข้อมูลที่ใช้นำมาจาก [Kaggle Data set](https://www.kaggle.com/shivamb/netflix-shows) และใน model นี้จะใช้เป็น [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)

Feature ที่สามารถเลือกได้ (ตอนนี้ปิดอยู่ เพราะระบบจะได้ไม่ต้องทำงานหนัก)
- Director : ผู้กำกับ
- Cast : นักแสดงนำ
- Listed in : ประเภท
- Description : เรื่องย่อ
'''
# เลือก feature
feature = st.multiselect('', ['director', 'cast', 'listed_in', 'description'],['director', 'cast', 'listed_in', 'description'])
list_of_feature = ['title']+feature
select_df = df[list_of_feature]
select_df.dropna(inplace=True)
st.write('ตัวอย่างชุดข้อมูลแสดงตาม feature ที่เลือก')
st.dataframe(select_df)

# รับค่า input จาก user
user_input = st.text_input("ใส่ชื่อหนัง", "The Matrix Revolutions")

if st.button("Get Recommend !!"):
    if user_input in bag_df.index:
        """**Movie Found !**"""
        st.write("Recommend movie for",user_input)
        recommendations(user_input)
    else:
        """
        **No movie found**

        Please enter movie with correct **Upper,Lower** case order

        example:

        Indiana Jones and the Kingdom of the Crystal Skull

        The Matrix Reloaded

        Naruto
        """

