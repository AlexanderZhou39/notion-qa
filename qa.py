"""Ask a question to the notion database."""
import faiss
from langchain import OpenAI
# from langchain.chains import VectorDBQAWithSourcesChain
from custom.chain import CustomChain
from custom.prompts import CUSTOM_COMBINE_PROMPT
from custom.validation import validate_answers
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
    llm=OpenAI(temperature=0), 
    combine_prompt=CUSTOM_COMBINE_PROMPT,
    vectorstore=store
)
result = chain({"question": args.question})
answers = result['answer'].split(':::')

print('validating answers...')
validated_result = validate_answers(result)
print('Validated:\n', validated_result, '\n\n')
print('Sources:\n', result['sources'])

# print(f"Answer: {result['answer']}")
# print(f"Sources: {result['sources']}")
