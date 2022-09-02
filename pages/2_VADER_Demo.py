import streamlit as st
import csv

st.set_page_config(
    page_title="Demo",
    page_icon="📝"
)

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
import pandas as pd
#demo
st.write("""
# VADER Sentiment Analysis
""")

tab1, tab2 = st.tabs(['Sentence', 'Files'])
#tab1
vas = tab1.text_input("Enter the word or sentence you want to do sentiment analysis.")
tab1.caption("This version is only for english")
pro = tab1.button("Process")
if pro:
    scr = sid.polarity_scores(vas)
    tab1.success(scr)
    cmp = scr["compound"]
    if cmp<=0.05 and cmp>=-0.05:
        tab1.success("Neutral")
    elif cmp>=0.05:
        tab1.success("Positive")
    elif cmp<=-0.05:
        tab1.success("Negative")
        
    rr = tab1.button("Again")
    if rr:
        st.experimental_rerun()

#tab2
def convert_df(data_files):
    return data_files.to_csv().encode('utf-8')
file = tab2.file_uploader("Upload File TSV", type=['tsv'])
if file is not None:
    data_files = pd.read_csv(file, sep='\t')
    tab2.dataframe(data_files)
    baris = []
    for row in data_files:
        baris.append(row)
    option = tab2.selectbox(
        'Select the name of the column you want to do sentiment analysis',
        (baris))
    prf = tab2.button('Process File')
    if prf:
        with st.spinner('Please Wait... Performing Analytical Calculations'):
            data_files['scores'] = data_files[option].apply(lambda content:sid.polarity_scores(data_files[option]))
        with st.spinner('Please Wait... Performing Labeling'):
            data_files['compound'] = data_files["scores"].apply(lambda score_dict:score_dict['compound'])
            data_files['label'] = data_files['compound'].apply(lambda c: 'Positive' if c>=0.05 else 'Negative' if c<=-0.05 else 'Neutral')
            tab2.dataframe(data_files)
        file_csv = convert_df(data_files)
        tab2.download_button(
            label="Download data as CSV",
            data = file_csv,
            file_name="vader.csv",
            mime='text/csv',
        )
else:
    tab2.caption('This version is only for english')
    tab2.write('You can convert your file into tsv. [Click Here](https://www.convertsimple.com/convert-csv-to-tsv/)')