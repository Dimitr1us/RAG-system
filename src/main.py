from google import genai
import numpy as np
import json
import os
import sys
from datetime import datetime
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Task import Task
from src.logger import logger

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

client = genai.Client(api_key=API_KEY)

with open("data/context.json", "r", encoding="utf-8") as f:
    data = json.load(f)

def cosine_similarity(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def clean_code(code):
    code = code.replace("```", "")
    code = code.replace("python", "")
    code = code.strip()
    return code

def run_solution(file_path, test_input, answer_input, name_function):
    with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()
    local_env = {}
    try:
        exec(code, {}, local_env)
        solve = local_env.get(name_function)
        k = 0
        for i in range(len(test_input)):
            answer = solve(test_input[i])
            k += (answer == answer_input[i])
        return round(k / len(answer_input), 2)
    except Exception:
        return 0.0

def askModel(something):
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=something
    )
    answer = response.candidates[0].content.parts[0].text
    return clean_code(answer)

def getEmbedding(something):
    context = client.models.embed_content(
        model="gemini-embedding-001",
        contents=something
    )
    return context.embeddings[0].values

def bestContext(prompt, k=3):
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
        with open("data/context.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    return [item for _, item in scored[:k]]

def Research(task):
    start_time = time.time()
    context = bestContext(task.Description(), 3)
    text = ""
    for item in context:
        text += item['task'] + "\n" + item['solution'] + "\n\n"

    prompt_without = task.Prompt()
    prompt_with = prompt_without + "\nДля лучшего решения учти следующие примеры:\n" + text + "\nЕсли решение совпадает с контекстом — всё равно верни только код функции."

    answer_rag = askModel(prompt_with)
    with open("solutions/solution_with_rag.py", "w", encoding="utf-8") as f:
        f.write(answer_rag)

    answer_no_rag = askModel(prompt_without)
    with open("solutions/solution_without_rag.py", "w", encoding="utf-8") as f:
        f.write(answer_no_rag)

    acc_rag = run_solution("solutions/solution_with_rag.py", task.Tests(), task.Answer(), task.Function_Name())
    acc_no = run_solution("solutions/solution_without_rag.py", task.Tests(), task.Answer(), task.Function_Name())

    elapsed = time.time() - start_time

    print(f"Solution without RAG: {acc_no:.2f}")
    print(f"Solution with RAG:    {acc_rag:.2f}")
    print(f"Time: {elapsed:.1f} sec")

    logger.info("Результаты решения", extra={
        "task": task.Description(),
        "function_name": task.Function_Name(),
        "accuracy_without_rag": acc_no,
        "accuracy_with_rag": acc_rag,
        "context_items": len(context),
        "elapsed_time": round(elapsed, 4),
        "timestamp": datetime.now().isoformat()
    })

def main():
    tasks = [
        Task("Напиши функцию, которая ищет максимальный элемент массива.", "solve", [[1,2,3],[1,2],[4,2,3,1]], [3,2,4]),
        Task("Напиши функцию, которая возвращает сумму всех элементов списка.", "solve", [[1,2,3],[10,20],[0]], [6,30,0]),
        Task("Напиши функцию, которая проверяет, является ли число простым.", "solve", [7,10,13,1], [True, False, True, False]),
    ]
    for t in tasks:
        print(f"\n=== Задача: {t.Description()}")
        Research(t)

if __name__ == "__main__":
    main()