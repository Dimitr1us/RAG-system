import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.Task import Task
from src.rag_core import generate_with_rag

st.set_page_config(page_title="RAG vs No-RAG", layout="wide")
st.title("🔍 Сравнение RAG и обычной генерации кода с Gemini")

tab1, tab2 = st.tabs(["🚀 Тестирование", "📊 История"])

with tab1:
    st.subheader("Описание задачи")
    user_input = st.text_area(
        "Напишите задачу:", 
        height=150,
        placeholder="Напиши функцию, которая проверяет, является ли число простым..."
    )

    if st.button("🚀 Запустить сравнение RAG vs Без RAG", type="primary", use_container_width=True):
        if not user_input.strip():
            st.warning("Введите описание задачи")
        else:
            with st.spinner("Генерация решений..."):
                result = generate_with_rag(user_input)

                st.success(f"Готово за {result['time']} секунд (использовано {result['context_items']} примеров)")

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("✅ С RAG")
                    st.code(result["code_rag"], language="python")
                with col2:
                    st.subheader("⚪️ Без RAG")
                    st.code(result["code_no_rag"], language="python")

with tab2:
    st.info("Здесь в будущем можно добавить историю запусков и статистику.")

st.caption("Курсовая работа • RAG-system")