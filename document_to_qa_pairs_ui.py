'''
streamlit run document_to_qa_pairs_ui.py --server.port 3073 
'''

import pandas as pd
import numpy as np
import requests
import re
import os
import time
import hashlib
import streamlit as st
import pandas as pd
from io import StringIO

md5 = lambda str2hash : hashlib.md5(str2hash.encode()).hexdigest()

st.title("Document to QA pair Converter")


def sentences_to_paragraphs(
    sentences
    ):

    sentence_size = len(sentences)

    paragraphs = []

    for i in range(0,sentence_size,step_size):
        start = i
        end = i + window_size
        if(end <= sentence_size+step_size):
            paragraph_sents = sentences[start:end]
            paragraph_sents = ' '.join(paragraph_sents)
            #print(f'{start} -> {end}: {paragraph_sents}\n\n')      
            paragraph_sents = re.sub(r'\n+', r' ', paragraph_sents)
            paragraphs.append(paragraph_sents)
            
    return paragraphs


system_prompt = f"""
Please generate question-answer pairs from the following document. Both the questions and answers should contain sufficient context information, and be self-contained, short, and abstract. The generated text is in the format of 

Q 1:
A 1:

Q 2:
A 2:

Q 3:
A 3:

...
"""


def paragragh_to_qa_pairs(
    paragraphs,
    model_api_url = 'http://37.224.68.132:27465/llm/llama_2_7b_batch',
    ):
    start = time.time()
    
    prompts = []
    for paragraph in paragraphs:  
        paragraph = paragraph.strip()
        prompt = f"""
[INST] <<SYS>> {system_prompt.strip()} <</SYS>>

Document: {paragraph}

[/INST] Sure. Here are the question and answer pairs: 
"""
        #print(f'prompt:{prompt}\n\n')
        prompts.append(prompt)
    
    response = requests.post(
    model_api_url,
    json = {
    "prompts": prompts,  
    "max_tokens": 1024,
    }
    ).json()

    running_time = time.time() - start

    #print(f'running time:\t{running_time:0.2f} s\n\n') 
    
    return response


def qa_pairs_text_to_qa_list(
    qa_pairs_text,
    ):
    qa_pairs = []

    for m in re.finditer(
        r'Q\s*\d{0,}\:\s*(?P<question>.*?\?)\s*\n+A\s*\d{0,}\:\s*(?P<answer>[^\n]+)',
        qa_pairs_text,
        flags = re.DOTALL
        ):

        qa_pair = m.groupdict()
        qa_pairs.append(qa_pair)
            
    return qa_pairs




uploaded_files = st.file_uploader(
    "Choose a file", 
    accept_multiple_files=False)

for uploaded_file in uploaded_files:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    string_data = stringio.read()
    st.write(f"{uploaded_file.name} uploaded. LLM is reading it to generate QA pairs.", )
    #st.write(string_data)

    start = time.time()
 
    sentences = requests.post(
    'http://37.224.68.132:27331/stanza/sentence_segmentation',
    json = {
    "text": string_data
    }).json()['sentences']

    window_size = 5
    step_size = 3 
    paragraphs = sentences_to_paragraphs(sentences)

    response1 = paragragh_to_qa_pairs(
        paragraphs,
        model_api_url = 'http://37.224.68.132:27465/llm/llama_2_7b_batch',
        )['responses']

    response2 = paragragh_to_qa_pairs(
        paragraphs,
        model_api_url = 'http://37.224.68.132:27464/llm/mistral_7b_batch',
        )['responses']

    ##
    data = []
    for p, q1, q2 in zip(
        paragraphs,
        response1,
        response2,
        ):
        qa1 = qa_pairs_text_to_qa_list(q1)
        qa2 = qa_pairs_text_to_qa_list(q2)
        for qa in qa1+qa2:
            qa["paragraph"] = p
            qa["file_name"] = uploaded_file.name
            data.append(qa)

    st.write(f"LLM generated {len(data)} QA pairs from {uploaded_file.name}. running time: {time.time() - start:0.2f} s", )
    
    ILLEGAL_CHARACTERS_RE = re.compile(r'[\000-\010]|[\013-\014]|[\016-\037]')

    df = pd.DataFrame(data).applymap(lambda x: ILLEGAL_CHARACTERS_RE.sub(r'', x) if isinstance(x, str) else x)
    df.to_excel(
        f"{uploaded_file.name}.xlsx",
        index = False,
        )


    with open(f"{uploaded_file.name}.xlsx", 'rb') as f:
        st.download_button(
        '📥 Download QA pairs',
        f,
        file_name= f"{uploaded_file.name}.xlsx")

