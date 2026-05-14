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


def save_to_context(task_description: str, solution_code: str, similarity_threshold: float = 0.85):
    """Сохраняет новую задачу в базу с семантической проверкой дубликатов"""
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            context = json.load(f)
        
        # Получаем эмбеддинг новой задачи
        new_embedding = get_embedding(task_description)
        
        for item in context:
            if "embedding" not in item or item["embedding"] is None:
                item["embedding"] = get_embedding(item["task"])
            
            similarity = cosine_similarity(new_embedding, item["embedding"])
            
            if similarity > similarity_threshold:
                return False, f"Слишком похожая задача уже существует (схожесть: {similarity:.3f})"
        
        new_item = {
            "task": task_description,
            "solution": solution_code,
            "embedding": new_embedding.tolist()
        }
        
        context.append(new_item)
        
        with open(DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(context, f, ensure_ascii=False, indent=4)
        
        return True, "Задача успешно добавлена в базу знаний"
        
    except Exception as e:
        return False, f"Ошибка сохранения: {str(e)}"


def run_test(code: str, inputs: list, expecteds: list):
    """Тестирует код и возвращает детальную информацию"""
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

    context_items = best_context(task_description, 3)
    examples = "\n\n".join([f"Пример:\n{item['task']}\nРешение:\n{item['solution']}" for item in context_items])

    prompt_base = f"Напиши функцию на Python: {task_description}"
    prompt_with = prompt_base + "\n\nУчти примеры:\n" + examples + "\n\nВерни только чистый код функции."
    prompt_without = prompt_base + "\n\nВерни только чистый код функции."

    code_rag = ask_model(prompt_with)
    code_no_rag = ask_model(prompt_without)

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