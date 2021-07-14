import streamlit as st
import pandas as pd
import base64
import requests

st.title('Demo applications using Flax-Sentence-Tranformers')

select_model = st.selectbox('Please select a jax-sentence-transformer model.', [
    'V3 DistilRoBERTa Base',
    'V3 MPNet Base']
)

if select_model == 'V3 DistilRoBERTa Base':
    st.markdown("""
                This is a distilroberta-base model trained on all the dataset of the 1B+ train corpus. It was trained with the v3 setup. See data_config.json and train_script.py in this respository how the model was trained and which datasets have been used.
                """)


anchor = st.text_input(
    'Please enter here the main text you want to compare:'
)

if anchor:
    n_texts = st.sidebar.number_input(
        f'How many texts you want to compare with {anchor}?',
        value=2,
        min_value=2)

    inputs = []

    for i in range(n_texts):

        input = st.sidebar.text_input(f'Text {i+1}:')

        inputs.append(input)



api_base_url = 'http://127.0.0.1:8000/similarity'

if anchor:
    if st.sidebar.button('Tell me the similarity.'):
        res = requests.get(url = api_base_url, params = {'anchor': anchor, 'inputs': inputs})
        d = res.json()['dataframe']
        df = pd.DataFrame(d, columns=['inputs', 'score'])
        df = df.sort_values('score', ascending=False)

        st.write(df)




