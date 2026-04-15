from google import genai
import numpy as np
import json
import os
from Task import Task

API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key = API_KEY)

with open("context.json", "r", encoding="utf-8") as f:
    data = json.load(f)

def cosine_similarity(a, b):
    if (np.linalg.norm(a)==0 or np.linalg.norm(b)==0):
        return 0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def clean_code(code):
    code = code.replace("```","")
    code = code.replace("python","")
    code = code.strip()
    return code

def run_solution(file_path,test_input,answer_input, name_function):
    with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()

    local_env = {}
    try:
        exec(code,{},local_env)

        solve = local_env.get(name_function)
        
        k=0
        for i in range(len(test_input)):
            answer = solve(test_input[i])
            k += (answer==answer_input[i])
    except:
        raise Exception(f"""Function {name_function} not found.""")
    
    return k/len(answer_input)

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
    updated = False
    for item in data:
        if "embedding" not in item:
            updated = True
            item["embedding"] = getEmbedding(item["task"])
        score = cosine_similarity(vectorPrompt, item["embedding"])
        scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)

    if updated:
        with open("context.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii= False,indent = 4)

    return [item for _, item in scored[:k]]

def Research(task):
    context = bestContext(task.Description(), 3)

    text=""
    for item in context:
        text=text+item['task']+"\n"+item['solution']+"\n"

    prompt_without_rag = task.Prompt()
    prompt_with_rag = prompt_without_rag + "Для лучшего решения задания учти во внимание также данный код:\n" + text + "Если решение полностью совпадает с контекстом, то всё равно отправь код назад."

    # answer = askModel(prompt_with_rag)

    # with open("solution_with_rag.py","w",encoding="utf-8") as f:
    #     f.write(answer)

    # answer = askModel(prompt_without_rag)

    # with open("solution_without_rag.py","w",encoding="utf-8") as f:
    #     f.write(answer)

    # print("Statistics:\n")
    # print(f"""Solution without RAG: {run_solution("solution_without_rag.py",task.Tests(),task.Answer(),task.function_name)}""")
    # print(f"""Solution with RAG: {run_solution("solution_with_rag.py",task.Tests(),task.Answer(),task.function_name)}""")

def main():
    task_max = Task("Напиши функцию, которая ищет максимальный элемент массива.","solve",[[1,2,3],[1,2],[4,2,3,1]],[3,2,4])
    Research(task_max)

if (__name__=="__main__"):
    main()