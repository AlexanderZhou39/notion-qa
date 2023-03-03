import openai
from custom.prompts import validation_prompt

def validate_answers(result):
    answer = result['answer']
    docs = result['docs']

    contents = ""
    for doc in docs:
        contents += 'Content: ' + doc.page_content.strip() + '\n'
    
    formatted_prompt = validation_prompt.format(
        statements=answer,
        contents=contents
    )

    if "I don't know" in answer:
        return "I don't know"

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": formatted_prompt}
        ]
    )

    return completion.choices[0].message.content


