import streamlit as st 
import pandas as pd 


st.title("Summary of Data")




df =  pd.read_csv("new_df.csv")

st.write("This dataset contains customer sentiments expressed in various sources such as social media, review platforms, testimonials, and more. The dataset includes text, sentiment (positive or negative), source of the sentiment, date/time of the sentiment, user ID, location, and confidence score. The sentiments reflect customers' opinions and experiences with products, services, movies, music, books, restaurants, websites, customer support, and more.")
st.write("Df shape :",df.shape)
st.dataframe(df)



st.image("models/graph1.png")

st.image("models/graph2.png")