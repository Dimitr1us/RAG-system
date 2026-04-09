from google import genai
import numpy as np
import json
import os

API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key = API_KEY)

with open("context.json", "r", encoding="utf-8") as f:
    data = json.load(f)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def clean_code(code):
    code = code.replace("```","")
    code = code.replace("python","")
    code = code.strip()
    return code

#делает запрос
def askModel(something):
    response = client.models.generate_content(
        model = "gemini-3-flash-preview",
        contents = something
    )
    answer = response.candidates[0].content.parts[0].text
    answer = clean_code(answer)
    return answer

#делает запрос
def getEmbedding(something):
    context = client.models.embed_content(
        model="gemini-embedding-001",
        contents=something
    )
    return context.embeddings[0].values

#делает запрос
def bestContext(prompt,k=3):
    vectorPrompt = getEmbedding(prompt)
    scored = []
    for item in data:
        score = cosine_similarity(vectorPrompt, getEmbedding(item["task"]))
        scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:k]]

task = "Напиши функцию, которая ищет максимальный элемент массива. Назови получившуюся функцию solve"

context = bestContext(task,2)

text=""
for item in context:
    text=text+item['task']+"\n"+item['solution']+"\n"

prompt_with_rag = f"""
Реши задачу на питоне: {task}
Для лучшего решения задания учти во внимание также данный код:
{text}
Назад отправь только код самой задачи.
Если решение полностью совпадает с контекстом, то всё равно отправь код назад.
"""


prompt_without_rag = f"""
Реши задачу на питоне: {task}
Назад отправь только код самой задачи.
"""


def main():
    answer = askModel(prompt_with_rag)

    with open("solution_with_rag.py","w",encoding="utf-8") as f:
        f.write(answer)

    answer = askModel(prompt_without_rag)

    with open("solution_without_rag.py","w",encoding="utf-8") as f:
        f.write(answer)

if (__name__=="__main__"):
    main()