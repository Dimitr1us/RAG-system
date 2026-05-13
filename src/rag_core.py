import json
import os
import numpy as np
import time
from google import genai

API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

DATA_PATH = "data/context.json"
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)


def cosine_similarity(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def clean_code(code: str) -> str:
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0]
    elif "```" in code:
        code = code.split("```")[1].split("```")[0]
    return code.strip()


def get_embedding(text: str):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return response.embeddings[0].values


def best_context(prompt: str, k: int = 3):
    vector_prompt = get_embedding(prompt)
    scored = []
    updated = False

    for item in data:
        if "embedding" not in item:
            updated = True
            item["embedding"] = get_embedding(item["task"])
        score = cosine_similarity(vector_prompt, item["embedding"])
        scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)

    if updated:
        with open(DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    return [item for _, item in scored[:k]]


def ask_model(prompt: str) -> str:
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )
    return clean_code(response.candidates[0].content.parts[0].text)


def generate_with_rag(task_description: str):
    start_time = time.time()

    context_items = best_context(task_description, 3)
    examples = "\n\n".join([
        f"Пример задачи:\n{item['task']}\nРешение:\n{item['solution']}" 
        for item in context_items
    ])

    prompt_base = f"Напиши функцию на Python: {task_description}"
    
    prompt_with = prompt_base + "\n\nУчти следующие примеры и реши аналогично:\n" + examples + "\n\nВерни только чистый код функции."
    prompt_without = prompt_base + "\n\nВерни только чистый код функции."

    code_rag = ask_model(prompt_with)
    code_no_rag = ask_model(prompt_without)

    elapsed = time.time() - start_time

    return {
        "code_rag": code_rag,
        "code_no_rag": code_no_rag,
        "time": round(elapsed, 2),
        "context_items": len(context_items)
    }