import streamlit as st
import sys
import os
import json
import numpy as np
from datetime import datetime
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.Task import Task
from src.logger import logger

# ====================== Инициализация ======================
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error("❌ GEMINI_API_KEY не найден в переменных окружения!")
    st.stop()

from google import genai
client = genai.Client(api_key=API_KEY)

# Загрузка контекста
with open("data/context.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# ====================== Вспомогательные функции ======================
def cosine_similarity(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def clean_code(code):
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0]
    elif "```" in code:
        code = code.split("```")[1].split("```")[0]
    return code.strip()

def getEmbedding(text):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return response.embeddings[0].values

def bestContext(prompt, k=3):
    vector_prompt = getEmbedding(prompt)
    scored = []
    updated = False
    for item in data:
        if "embedding" not in item:
            updated = True
            item["embedding"] = getEmbedding(item["task"])
        score = cosine_similarity(vector_prompt, item["embedding"])
        scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    if updated:
        with open("data/context.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    return [item for _, item in scored[:k]]

def askModel(prompt):
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )
    return clean_code(response.candidates[0].content.parts[0].text)

# ====================== Streamlit ======================
st.set_page_config(page_title="RAG vs No-RAG", layout="wide")
st.title("🔍 Сравнение RAG и обычной генерации кода с Gemini")

tab1, tab2 = st.tabs(["🚀 Тестирование", "📊 История"])

with tab1:
    st.subheader("Описание задачи")
    user_input = st.text_area("Напишите задачу для генерации функции:", 
                              height=150,
                              placeholder="Напиши функцию, которая проверяет, является ли число простым...")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        run_button = st.button("Запустить сравнение RAG vs No-RAG", type="primary", use_container_width=True)
    
    if run_button and user_input:
        with st.spinner("Генерируем решения..."):
            start_time = time.time()
            
            # Создаём задачу
            task = Task(user_input, "solve", [], [])  # тесты можно добавить позже
            
            context_items = bestContext(task.Description(), 3)
            examples = ""
            for item in context_items:
                examples += item['task'] + "\n" + item['solution'] + "\n\n"

            prompt_without = task.Prompt()
            prompt_with = prompt_without + "\nУчти следующие примеры и реши аналогично:\n" + examples

            # Генерация
            code_rag = askModel(prompt_with)
            code_no_rag = askModel(prompt_without)

            st.success(f"Готово за {time.time()-start_time:.1f} сек")

            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("С RAG")
                st.code(code_rag, language="python")
            with col_b:
                st.subheader("Без RAG")
                st.code(code_no_rag, language="python")

with tab2:
    st.info("Здесь позже будет история запусков из логов")

st.caption("Курсовая работа — Сравнительный анализ Retrieval-Augmented Generation")