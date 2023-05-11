# Run chatGPT with internal confluence pages.

Ref: https://medium.com/@francois.ascani/running-chatgpt-on-your-internal-confluence-documentation-d7761aa8fc68 

**NOTES: Not fully functional. I can get some questions answered, but for most questions, the bot answers "I don't know"**

### Usages

* get the OpenAI API Key

ref: https://platform.openai.com/account/api-keys

* get atlassian access key

[create first a personal API token (PAT)](https://confluence.atlassian.com/enterprise/using-personal-access-tokens-1026032365.html)

Account profile -> Settings -> Password -> Create and manage API tokens

* update environment variable

```
$ cp .env.sample .env

# update all keys in .env
```

run a test to get confluence pages
```
$ virtualenv env
$ source env/bin/activate

$ pip install -r requirements.txt
python ./get_pages.py
```
If successful, you will see several new files are generated

```
output_100.json
output_all.json
DOC_title_content_embeddings.csv
```

Run the application
```
$ virtualenv env
$ source env/bin/activate
$ flask run
...
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
```

Wait for a while (60+ seconds), you can access the website locally

Access the application via http://127.0.0.1:5000

![index](images/index.png)
