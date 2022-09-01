import streamlit as st

st.set_page_config(
    page_title="Argumentation",
    page_icon="📊"
)

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd
#from sklearn.metrics import classification_report
#cr = classification_report
import matplotlib.pyplot as plts

sid = SentimentIntensityAnalyzer()

#intro
st.write("""
# Argumentation VADER Sentiment Analysis
In using VADER for this final project using review data from the Google Play Store
""")
st.write("This is review data from the ZOOM application which was scraped using Google Collaboratory")
df = pd.read_csv('Data Ulasan.tsv', sep='\t')
st.dataframe(df)

#implementation
st.write("""
# Implementation
This is review data that has been implemented using VADER
""")
df['scores'] = df['content'].apply(lambda content:sid.polarity_scores(content))
st.dataframe(df)
st.write("The results obtained are compound, negative, neutral and positive values")

#labeling
st.write("""
# Labeling
In this table, sentiment analysis has been applied using VADER and labeled according to the compound value.
""")
df['compound'] = df['scores'].apply(lambda score_dict:score_dict['compound'])
df['label'] = df['compound'].apply(lambda c: 'P' if c>=0.05 else 'N' if c<=-0.05 else 'NT')
st.dataframe(df)
st.markdown("""
Explanation:
- N = Negative
- P = Positive
- NT = Neutral
""")
st.write("The following are the number of positive, negative and neutral results from labeling results using VADER")
st.table(df['label'].value_counts())

#manual
st.write("""
# Manual Labeling
In the table below the results of the VADER labeling which have been added with manual labeling carried out by the researchers themselves.
""")
df1 = pd.read_csv('vader/data/CM.tsv', sep='\t')
st.dataframe(df1)
st.markdown("""
Explanation:
- N = Negative
- P = Positive
- NT = Neutral
""")
st.write("The following are the number of positive, negative and neutral results from manual labeling")
st.table(df1['label_manual'].value_counts())

#chart
st.write("""
# Comparison
In the graph below shows a comparison of the results of labeling using VADER and manual labeling.
""")
c1, c2 = st.columns(2)
c1.write("VADER Labeling")
c1.bar_chart(df['label'].value_counts())
c2.write("Manual Labeling")
c2.bar_chart(df1['label_manual'].value_counts())
y_mnl = df1['label_manual']
y_vd = df1['label_vader']
st.write("The following is the result of a cross-comparison between manual labeling and labeling using VADER")
st.table(pd.crosstab(y_mnl, y_vd))
#st.write("The following accuracy results are generated")
#st.code(cr(y_mnl,y_vd))

#conc
st.write("""
# Conclusion
Based on the accuracy results obtained as much as 0.85 or 85%, it shows that VADER can perform sentiment analysis on a sentence or word with very good accuracy and speed.
""")
