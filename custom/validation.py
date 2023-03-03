import openai
from custom.prompts import validation_prompt

def validate_answers(result):
    answer = result['answer']
    docs = result['docs']

    contents = ""
    for doc in docs:
        contents += 'CONTENT: ' + doc.page_content + '\n'
    
    formatted_prompt = validation_prompt.format(
        statements=answer,
        contents=contents
    )

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[{"role": "user", "content": formatted_prompt}]
    )

    return completion.choices[0].message.content


