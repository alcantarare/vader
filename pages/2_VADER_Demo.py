import streamlit as st

st.set_page_config(
    page_title="Demo",
    page_icon="📝"
)

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

#demo
st.write("""
# VADER Sentiment Analysis
""")
vas = st.text_input("Enter the word or sentence you want to do sentiment analysis.")
pro = st.button("Process")

if pro:
    scr = sid.polarity_scores(vas)
    st.success(scr)
    cmp = scr["compound"]
    if cmp>=0.05:
        st.success("Positive")
    elif cmp<=0.05:
        st.success("Negative")
    else:
        st.success("Neutral")
    rr = st.button("Again")
    if rr:
        st.experimental_rerun()
        