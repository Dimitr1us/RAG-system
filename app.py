import streamlit as st
import sys
import os
import ast

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag_core import generate_with_rag

st.set_page_config(page_title="RAG vs No-RAG", layout="wide")
st.title("🔍 Сравнение RAG и обычной генерации кода с Gemini")

tab1, tab2 = st.tabs(["🚀 Тестирование", "📊 История"])

with tab1:
    st.subheader("Описание задачи")
    user_input = st.text_area("Напишите задачу:", height=110, placeholder="Напиши функцию, которая...")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Тестовые входные данные")
        test_input_str = st.text_area("Пример: 1,2,3 или [1,2],[3,4]", height=100, placeholder="1,2,3\n[1,2],[3,4]")
    
    with col2:
        st.subheader("Ожидаемые результаты")
        expected_str = st.text_area("Должно быть столько же, сколько входных данных", height=100)

    if st.button("🚀 Запустить сравнение RAG vs Без RAG", type="primary", use_container_width=True):
        if not user_input.strip():
            st.warning("Введите описание задачи")
            st.stop()

        # Парсинг тестов
        inputs = []
        expecteds = []
        parse_error = None

        try:
            if test_input_str.strip():
                inputs = [ast.literal_eval(item.strip()) for item in test_input_str.split(',') if item.strip()]
        except Exception as e:
            parse_error = f"Ошибка парсинга входных данных: {e}"

        try:
            if expected_str.strip():
                expecteds = [ast.literal_eval(item.strip()) for item in expected_str.split(',') if item.strip()]
        except Exception as e:
            parse_error = f"Ошибка парсинга ожидаемых результатов: {e}"

        with st.spinner("Генерация и тестирование..."):
            result = generate_with_rag(user_input, inputs, expecteds)

            st.success(f"Готово за {result['time']} сек")

            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("✅ С RAG")
                st.code(result["code_rag"], language="python")
                if "accuracy_rag" in result:
                    if result["error_rag"]:
                        st.error(result["error_rag"])
                    else:
                        st.success(f"Точность: **{result['accuracy_rag']:.1%}**")

            with col_b:
                st.subheader("⚪️ Без RAG")
                st.code(result["code_no_rag"], language="python")
                if "accuracy_no_rag" in result:
                    if result["error_no_rag"]:
                        st.error(result["error_no_rag"])
                    else:
                        st.success(f"Точность: **{result['accuracy_no_rag']:.1%}**")

st.caption("Курсовая работа • RAG-system")