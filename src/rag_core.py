import json
import os
import numpy as np
import time
from google import genai

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found")

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


def save_to_context(task_desc: str, solution_code: str, similarity_threshold: float = 0.85):
    try:
        import json
        import os
        from datetime import datetime

        path = "data/context.json"
        os.makedirs("data", exist_ok=True)

        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = []

        new_embedding = get_embedding(task_desc)
        if isinstance(new_embedding, (list, tuple)):
            new_embedding = list(new_embedding)
        elif hasattr(new_embedding, 'tolist'):
            new_embedding = new_embedding.tolist()
        else:
            new_embedding = list(new_embedding)

        max_similarity = 0.0
        similar_task = None

        for item in data:
            if "embedding" in item:
                existing_emb = item["embedding"]
                similarity = cosine_similarity(new_embedding, existing_emb)
                if similarity > max_similarity:
                    max_similarity = similarity
                    similar_task = item.get("task", "")[:100]

        if max_similarity > similarity_threshold:
            return False, (f"Задача слишком похожа на уже существующую "
                          f"(сходство: {max_similarity:.3f}).\n"
                          f"Похожая задача: {similar_task}...")

        new_entry = {
            "task": task_desc.strip(),
            "solution": solution_code.strip(),
            "embedding": new_embedding
        }

        data.append(new_entry)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        return True, f"✅ Задача успешно добавлена! (Сходство с ближайшей: {max_similarity:.3f})"

    except Exception as e:
        return False, f"Ошибка при сохранении: {str(e)}"


def run_test(code: str, inputs: list, expecteds: list):
    try:
        local_env = {}
        exec(code, {}, local_env)
        
        func = local_env.get("solve")
        if not func:
            for name, obj in local_env.items():
                if callable(obj) and not name.startswith("__"):
                    func = obj
                    break
                    
        if not func:
            return 0.0, "Не удалось найти функцию", None

        results = []
        correct = 0
        
        for i, (inp, exp) in enumerate(zip(inputs, expecteds)):
            try:
                result = func(inp)
                is_correct = result == exp
                if is_correct:
                    correct += 1
                
                results.append({
                    "test_num": i + 1,
                    "input": inp,
                    "expected": exp,
                    "actual": result,
                    "correct": is_correct
                })
            except Exception as e:
                results.append({
                    "test_num": i + 1,
                    "input": inp,
                    "expected": exp,
                    "actual": f"ERROR: {e}",
                    "correct": False
                })
                return 0.0, f"Ошибка на тесте {i+1}", results
        
        accuracy = correct / len(inputs) if inputs else 0.0
        return accuracy, None, results

    except Exception as e:
        return 0.0, f"Ошибка выполнения кода: {str(e)}", None


def generate_with_rag(task_description: str, test_inputs=None, expected_outputs=None):
    start_time = time.time()

    context_items = best_context(task_description, k=3)
    
    examples_text = "\n\n".join([
        f"Пример {i+1}:\nЗадача: {item['task'].strip()}\nРешение:\n{item['solution'].strip()}" 
        for i, item in enumerate(context_items)
    ])

    # === Улучшенный system prompt ===
    system_instruction = """Ты — сильный Python-разработчик и алгоритмист. 
    Напиши корректную, чистую и эффективную функцию по описанию задачи.
    - Всегда называй функцию `solve`, если явно не указано другое.
    - Возвращай **только код** функции без markdown, без объяснений.
    - Можно импортировать модули внутри функции при необходимости."""

    # Промпт с RAG
    prompt_rag = f"""{system_instruction}

    Примеры похожих задач:

    {examples_text}

    Задача:
    {task_description}

    Решение:"""

    prompt_no_rag = f"""{system_instruction}

    Задача:
    {task_description}

    Решение:"""

    code_rag = ask_model(prompt_rag)
    code_no_rag = ask_model(prompt_no_rag)
    

    result = {
        "code_rag": code_rag,
        "code_no_rag": code_no_rag,
        "time": round(time.time() - start_time, 2),
        "context_items": len(context_items)
    }

    if test_inputs and expected_outputs and len(test_inputs) == len(expected_outputs):
        acc_rag, err_rag, details_rag = run_test(code_rag, test_inputs, expected_outputs)
        acc_no, err_no, details_no = run_test(code_no_rag, test_inputs, expected_outputs)
        
        result["accuracy_rag"] = acc_rag
        result["accuracy_no_rag"] = acc_no
        result["error_rag"] = err_rag
        result["error_no_rag"] = err_no
        result["test_details_rag"] = details_rag
        result["test_details_no"] = details_no

    return result