#!/usr/bin/env python3

import os
from atlassian import Confluence # See https://atlassian-python-api.readthedocs.io/index.html
from dotenv import load_dotenv
import json

load_dotenv()

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

