"""Ask a question to the notion database."""
import faiss
from langchain import OpenAI
# from langchain.chains import VectorDBQAWithSourcesChain
from custom.chain import CustomChain
from custom.prompts import CUSTOM_COMBINE_PROMPT
from custom.validation import validate_answers
from custom.llm import CustomOpenAI
import pickle
import argparse

parser = argparse.ArgumentParser(description='Ask a question to the notion DB.')
parser.add_argument('question', type=str, help='The question to ask the notion DB')
args = parser.parse_args()

# Load the LangChain.
index = faiss.read_index("docs.index")

with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)

store.index = index
print('running initial prompt...')
chain = CustomChain.from_llm(
    llm=CustomOpenAI(temperature=0, model_name="gpt-3.5-turbo"), 
    combine_prompt=CUSTOM_COMBINE_PROMPT,
    vectorstore=store
)
result = chain({"question": args.question})
answers = result['answer'].split(':::')
# for doc in result['docs']:
#     print(doc.page_content)

print('\n\nvalidating answers...')
validated_result = validate_answers(result)
print('\n\nOriginal:\n', result['answer'].strip(), '\n')
print('Validated:\n', validated_result, '\n')
print('Sources:\n', result['sources'])

# print(f"Answer: {result['answer']}")
# print(f"Sources: {result['sources']}")
