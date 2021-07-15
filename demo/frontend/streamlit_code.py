import streamlit as st
import pandas as pd
import base64
import requests

st.title('Demo using Flax-Sentence-Tranformers')

st.sidebar.title('')

st.markdown('''

Hi! This is the demo for the [flax sentence embeddings](https://huggingface.co/flax-sentence-embeddings) created for the **Flax/JAX community week 🤗**. We are going to use three flax-sentence-embeddings models: a **distilroberta base**, a **mpnet base** and a **minilm-l6**. All were trained on all the dataset of the 1B+ train corpus with the v3 setup.

---

**Instructions**: You can compare the similarity of a main text with other texts of your choice (in the sidebar). In the background, we'll create an embedding for each text, and then we'll use the cosine similarity function to calculate a similarity metric between our main sentence and the others.

For more cool information on sentence embeddings, see the [sBert project](https://www.sbert.net/examples/applications/computing-embeddings/README.html).

Please enjoy!!
''')


anchor = st.text_input(
    'Please enter here the main text you want to compare:'
)

if anchor:
    n_texts = st.sidebar.number_input(
        f'''How many texts you want to compare with: '{anchor}'?''',
        value=2,
        min_value=2)

    inputs = []

    for i in range(n_texts):

        input = st.sidebar.text_input(f'Text {i+1}:')

        inputs.append(input)



api_base_url = 'http://127.0.0.1:8000/similarity'

if anchor:
    if st.sidebar.button('Tell me the similarity.'):
        res_distilroberta = requests.get(url = api_base_url, params = dict(anchor = anchor,
                                                                           inputs = inputs,
                                                                           model = 'distilroberta'))
        res_mpnet = requests.get(url = api_base_url, params = dict(anchor = anchor,
                                                                   inputs = inputs,
                                                                   model = 'mpnet'))
        res_minilm_l6 = requests.get(url = api_base_url, params = dict(anchor = anchor,
                                                                       inputs = inputs,
                                                                       model = 'minilm_l6'))

        d_distilroberta = res_distilroberta.json()['dataframe']
        d_mpnet = res_mpnet.json()['dataframe']
        d_minilm_l6 = res_minilm_l6.json()['dataframe']

        index = list(d_distilroberta['inputs'].values())
        df_total = pd.DataFrame(index=index)
        df_total['distilroberta'] = list(d_distilroberta['score'].values())
        df_total['mpnet'] = list(d_mpnet['score'].values())
        df_total['minilm_l6'] = list(d_minilm_l6['score'].values())

        st.write('Here are the results for our three models:')
        st.write(df_total)
        st.write('Visualize the results of each model:')
        st.area_chart(df_total)


