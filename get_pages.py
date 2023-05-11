#!/usr/bin/env python3

import os
import json
from app import connect_to_Confluence,get_all_pages,collect_title_body_embeddings
from dotenv import load_dotenv

load_dotenv()

confluence=connect_to_Confluence()
pages=get_all_pages(confluence, space=os.environ["atlassian-space"])

# Get all datas
json_string = json.dumps(pages)
with open("output_all.json", "w") as file:
    # Convert the list to a string using the join() method
    file.write(json_string)

# Get first 100 records for test purpose
pages100=pages[:100]
json_string = json.dumps(pages100)
with open("output_100.json", "w") as file:
    # Convert the list to a string using the join() method
    file.write(json_string)

# Extract title, body and number of tokens
# If successful, file named "DOC_title_content_embeddings.csv" will be created locally
collect_title_body_embeddings(pages, save_csv=True)

# For test only and save the cost of OpenAI API, you can feed only 100 records 
# collect_title_body_embeddings(pages100, save_csv=True)
