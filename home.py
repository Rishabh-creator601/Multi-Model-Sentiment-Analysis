import streamlit as st
from loader  import extract_score,sentiment_analyzer,vader_model,analyze
import pandas as pd 

import nltk 
nltk.downloader.download('vader_lexicon')

st.title("Multi Model Sentiment Evaluation")

# Init state
if "default" not in st.session_state:
    st.session_state["default"] = ""

text_options = [
    "I love this product!",
    "The service was terrible.",
    "This movie is amazing!",
    "Just had the best meal of my life!",
    "The product arrived damaged.",
]

# Pills FIRST
selection = st.pills(
    "Some Examples",
    text_options,
    selection_mode="single"
)

# Update state BEFORE text_area renders
if selection:
    st.session_state["default"] = selection

st.write("Enter Something to proceed ....")

# Text area AFTER state update
txt = st.text_area(
    label="Input Text",
    label_visibility="collapsed",
    placeholder="Enter something here...",
    value=st.session_state["default"]
)


if st.button("Analyze sentiment 💠"):
    vader_score =  extract_score(vader_model,txt)
    textblob_score =  sentiment_analyzer(txt)
    nb_score,rf_score,xgb_score =  analyze(txt)
    
    scores =  [vader_score,textblob_score,nb_score,rf_score,xgb_score]
    
    score_df =  pd.DataFrame({
        
        "Model " :["Vader Sentiment","Text Blob ","Gaussain NB","RandomForest","XGBoost"],
        "Scores": scores
        
        })
    
    
    st.dataframe(score_df)
    
    
    avg =  sum(scores)/len(scores)
    if avg > 0.5:
        st.markdown("## Overall Review : Positive ")
    else:
        st.markdown("## Overall Review : Negative ")
        