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

def bestContext(prompt):
    vectorPrompt = getEmbedding(prompt)
    bestTask = data[0]["task"]
    bestSolution = data[0]["solution"]
    bestVector = getEmbedding(bestTask)
    for item in data:
        currentTaskVector = getEmbedding(item["task"])
        if (cosine_similarity(vectorPrompt,currentTaskVector)>cosine_similarity(vectorPrompt,bestVector)):
            bestTask = item["task"]
            bestSolution = item["solution"]
            bestVector = currentTaskVector
    return (bestTask,bestSolution, bestVector)

task = "Напиши функцию, которая ищет максимальный"

context = bestContext(task)

prompt = f"""
Реши задачу на питоне: {task}
Для лучшего решения задания учти во внимание также данный код:
{context[1]}
Назад отправь только код самой задачи.
Если решение полностью совпадает с контекстом, то всё равно отправь код назад.
"""

print(askModel(prompt))


