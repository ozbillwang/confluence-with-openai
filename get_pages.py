#!/usr/bin/env python3

import os
import json
from confluencepages import connect_to_Confluence, get_all_pages
from dotenv import load_dotenv

load_dotenv()

confluence=connect_to_Confluence()
pages=get_all_pages(confluence, space='CCOE')

# Get all datas
json_string = json.dumps(pages)
with open("output_all.json", "w") as file:
    # Convert the list to a string using the join() method
    file.write(json_string)

# Get first 100 records for test purpose
json_string = json.dumps(pages[:100])
with open("output_100.json", "w") as file:
    # Convert the list to a string using the join() method
    file.write(json_string)
