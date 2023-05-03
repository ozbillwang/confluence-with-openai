#!/usr/bin/env python3

import os
from atlassian import Confluence # See https://atlassian-python-api.readthedocs.io/index.html
from dotenv import load_dotenv
import json

load_dotenv()

# Step 1
def get_url():
    # define the confluence url 'https://example.atlassian.net'
    return os.environ["atlassian-url"]

def connect_to_Confluence():
    '''
    Connect to Confluence

    We use the API token for the cloud
    To create an API token here: Confluence -> Profile Pic -> Settings -> Password -> Create and manage tokens

    Return
    ------
    A connector to Confluence
    '''

    url = get_url()
    username = os.environ["atlassian-username"]
    password = os.environ["atlassian-api-token"]

    confluence = Confluence(
        url=url,
        username=username,
        password=password,
        cloud=True)

    return confluence

# Step 2
def get_all_pages(confluence, space=os.environ["atlassian-space"]):
    '''
    Get all the pages within the space.

    Parameters
    ----------
    confluence: a connector to Confluence
    space: Space of the Confluence (i.e. 'MY-SPACE')

    Return
    ------
    List of page objects. Each page object has all the information concerning
    a Confluence page (title, body, etc)
    '''

    # There is a limit of how many pages we can retrieve one at a time
    # so we retrieve 100 at a time and loop until we know we retrieved all of
    # them.
    keep_going = True
    start = 0
    limit = 100
    pages = []
    while keep_going:
        results = confluence.get_all_pages_from_space(space, start=start, limit=100, status=None, expand='body.storage', content_type='page')
        pages.extend(results)
        if len(results) < limit:
            keep_going = False
        else:
            start = start + limit
    return pages

# Step 3
import nltk
from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
import openai
import numpy as np
import pandas as pd

# Set the API key
openai.api_key = os.environ["openai-api-key"]

def get_doc_model():
    '''
    Model string to calculate the embeddings.
    '''
    return 'text-search-curie-doc-001'

def get_embeddings(text: str, model: str) -> list[float]:
    '''
    Calculate embeddings.

    Parameters
    ----------
    text : str
        Text to calculate the embeddings for.
    model : str
        String of the model used to calculate the embeddings.

    Returns
    -------
    list[float]
        List of the embeddings
    '''
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def get_max_num_tokens():
    return 2046

def collect_title_body_embeddings(pages, save_csv=True):
    '''
    From a list of page objects, get the title and the body, calculate
    the number of tokens as well as the embeddings of the body.

    Parameters
    ----------
    pages: List of page objects, i.e. output of get_all_pages()
    save_csv: Boolean. If True, the dataframe is saved locally
    into a CSV file.

    Return
    ------
    A dataframe of the title and body of all pages.
    '''

    collect = []
    for page in pages:
        title = page['title']
        link = get_url() + '/wiki/spaces/MY-SPACE/pages/' + page['id']
        htmlbody = page['body']['storage']['value']
        htmlParse = BeautifulSoup(htmlbody, 'html.parser')
        body = []
        for para in htmlParse.find_all("p"):
            # Keep only a sentence if there is a subject and a verb
            # Otherwise, we assume the sentence does not contain enough useful information
            # to be included in the context for openai
            sentence = para.get_text()
            tokens = nltk.tokenize.word_tokenize(sentence)
            token_tags = nltk.pos_tag(tokens)
            tags = [x[1] for x in token_tags]
            if any([x[:2] == 'VB' for x in tags]): # There is at least one verb
                if any([x[:2] == 'NN' for x in tags]): # There is at least noun
                    body.append(sentence)
        body = '. '.join(body)
        # Calculate number of tokens
        tokens = tokenizer.encode(body)
        collect += [(title, link, body, len(tokens))]
    DOC_title_content_embeddings = pd.DataFrame(collect, columns=['title', 'link', 'body', 'num_tokens'])
    # Caculate the embeddings
    # Limit first to pages with less than 2046 tokens
    DOC_title_content_embeddings = DOC_title_content_embeddings[DOC_title_content_embeddings.num_tokens<=get_max_num_tokens()]
    doc_model = get_doc_model()
    DOC_title_content_embeddings['embeddings'] = DOC_title_content_embeddings.body.apply(lambda x: get_embeddings(x, doc_model))

    if save_csv:
        DOC_title_content_embeddings.to_csv('DOC_title_content_embeddings.csv', index=False)

    return DOC_title_content_embeddings

# Step 4
def update_internal_doc_embeddings():
    # Connect to Confluence
    confluence = connect_to_Confluence()
    # Get page contents
    pages = get_all_pages(confluence, space='MY-SPACE')
    # Extract title, body and number of tokens
    DOC_title_content_embeddings= collect_title_body_embeddings(pages, save_csv=True)
    return DOC_title_content_embeddings

# Step 5
import numpy as np

def vector_similarity(x, y):
    return np.dot(np.array(x), np.array(y))

# Step 6
def order_document_sections_by_query_similarity(query: str, doc_embeddings: pd.DataFrame):
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections.

    Return the list of document sections, sorted by relevance in descending order.
    """
    query_model = get_query_model()
    query_embedding = get_embeddings(query, model=query_model)
    doc_embeddings['similarity'] = doc_embeddings['embeddings'].apply(lambda x: vector_similarity(x, query_embedding))
    doc_embeddings.sort_values(by='similarity', inplace=True, ascending=False)
    doc_embeddings.reset_index(drop=True, inplace=True)

    return doc_embeddings

# Step 7
def construct_prompt(query, doc_embeddings):

    MAX_SECTION_LEN = get_max_num_tokens()
    SEPARATOR = "\n* "
    separator_len = len(tokenizer.tokenize(SEPARATOR))

    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_links = []

    for section_index in range(len(doc_embeddings)):
        # Add contexts until we run out of space.
        document_section = doc_embeddings.loc[section_index]

        chosen_sections_len += document_section.num_tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break

        chosen_sections.append(SEPARATOR + document_section.body.replace("\n", " "))
        chosen_sections_links.append(document_section.link)

    header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
    prompt = header + "".join(chosen_sections) + "\n\n Q: " + query + "\n A:"

    return (prompt,  chosen_sections_links)

# Step 8
def internal_doc_chatbot_answer(query, DOC_title_content_embeddings):

    # Order docs by similarity of the embeddings with the query
    DOC_title_content_embeddings = order_document_sections_by_query_similarity(query, DOC_title_content_embeddings)
    # Construct the prompt
    prompt, links = construct_prompt(query, DOC_title_content_embeddings)
    # Ask the question with the context to ChatGPT
    COMPLETIONS_MODEL = "text-davinci-002"

    response = openai.Completion.create(
        prompt=prompt,
        temperature=0,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        model=COMPLETIONS_MODEL
    )

    output = response["choices"][0]["text"].strip(" \n")

    return output, links

# Step 9
import os
from flask import Flask, request, render_template
import datetime
import pandas as pd

# app = Flask(__name__, template_folder=os.path.join(path_root, 'internal-doc-chatbot'))
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        text_input = request.form['text_input']
        text_output, links = process_text(text_input)
        print(text_output)
        return render_template('index.html', text_output=text_output, links=links)
    return render_template('index.html')

def parse_numbers(s):
  return [float(x) for x in s.strip('[]').split(',')]

def return_Confluence_embeddings():
    # Today's date
    today = datetime.datetime.today()
    # Current file where the embeddings of our internal Confluence document is saved
    Confluence_embeddings_file = 'DOC_title_content_embeddings.csv'
    # Run the embeddings again if the file is more than a week old
    # Otherwise, read the save file
    Confluence_embeddings_file_date = datetime.datetime.fromtimestamp(os.path.getmtime(Confluence_embeddings_file))
    delta = today - Confluence_embeddings_file_date
    if delta.days > 7:
        DOC_title_content_embeddings= update_internal_doc_embeddings()
    else:
        DOC_title_content_embeddings= pd.read_csv(Confluence_embeddings_file, dtype={'embeddings': object})
        DOC_title_content_embeddings['embeddings'] = DOC_title_content_embeddings['embeddings'].apply(lambda x: parse_numbers(x))

    return DOC_title_content_embeddings

def process_text(query):

    DOC_title_content_embeddings= return_Confluence_embeddings()
    output, links = internal_doc_chatbot_answer(query, DOC_title_content_embeddings)

    return output, links

if __name__ == '__main__':
    app.run()
