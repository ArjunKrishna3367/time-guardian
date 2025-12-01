from dotenv import load_dotenv
load_dotenv()
import os

api_key = os.getenv('GOOGLE_API_KEY')
import google.generativeai as genai
genai.configure(api_key=api_key)

import pprint
for model in genai.list_models():
    pprint.pprint(model)