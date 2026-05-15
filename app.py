import streamlit as st
import sys
import os
import ast
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag_core import generate_with_rag, save_to_context

st.set_page_config(
    page_title="RAG vs No-RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔍 RAG vs No-RAG — Генерация кода на Python с Gemini")
st.markdown("**Курсовая работа** • Сравнение эффективности Retrieval-Augmented Generation")

with st.sidebar:
    st.header("Настройки")
    api_key = st.text_input("GEMINI_API_KEY", type="password", 
                           value=os.getenv("GEMINI_API_KEY", ""))
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
    st.caption("Модель: gemini-3-flash-preview")

tab1, tab2 = st.tabs(["🚀 Тестирование", "📚 База знаний"])

# ====================== TAB 1: Тестирование ======================
with tab1:
    st.subheader("Описание задачи")
    user_input = st.text_area(
        "Введите описание задачи", 
        height=100,
        placeholder="Напиши функцию, которая ищет максимальный элемент массива...",
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Тестовые входные данные")
        test_input_str = st.text_area(
            "Каждая строка — один тест", 
            height=160,
            placeholder="[[1,2,3]]\n[[1,2]]\n[[4,2,3,1]]"
        )
    with col2:
        st.subheader("Ожидаемые результаты")
        expected_str = st.text_area(
            "Каждая строка — один результат", 
            height=160,
            placeholder="3\n2\n4"
        )

    if st.button("🚀 Запустить сравнение RAG vs Без RAG", type="primary", use_container_width=True):
        if not user_input.strip():
            st.warning("Введите описание задачи")
            st.stop()

        def parse_test_line(line: str):
            line = line.strip()
            if not line:
                return None
            try:
                return ast.literal_eval(line)
            except:
                return line

        inputs = [parse_test_line(line) for line in test_input_str.split('\n') if parse_test_line(line) is not None]
        expecteds = [parse_test_line(line) for line in expected_str.split('\n') if parse_test_line(line) is not None]

        if len(inputs) == 0 or len(inputs) != len(expecteds):
            st.error("Проверьте тестовые данные и ожидаемые результаты")
            st.stop()

        with st.status("⏳ Генерация решений... Это может занять 15–40 секунд", expanded=True) as status:
            try:
                result = generate_with_rag(user_input, inputs, expecteds)

                status.update(label="✅ Готово!", state="complete")

                st.success(f"Готово за {result['time']} секунд • Примеров из базы: {result.get('context_items', 0)}")

                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("✅ С RAG")
                    st.code(result["code_rag"], language="python")
                    if "accuracy_rag" in result:
                        if result.get("error_rag"):
                            st.error(result["error_rag"])
                        else:
                            st.metric("Точность (RAG)", f"{result['accuracy_rag']:.1%}")

                with c2:
                    st.subheader("⚪️ Без RAG")
                    st.code(result["code_no_rag"], language="python")
                    if "accuracy_no_rag" in result:
                        if result.get("error_no_rag"):
                            st.error(result["error_no_rag"])
                        else:
                            st.metric("Точность (No RAG)", f"{result['accuracy_no_rag']:.1%}")

                if result.get("test_details_rag"):
                    st.subheader("📋 Подробные результаты тестов")
                    for det_rag, det_no in zip(result["test_details_rag"], result["test_details_no"]):
                        with st.expander(f"Тест {det_rag['test_num']}: {det_rag['input']}"):
                            col_r, col_n = st.columns(2)
                            with col_r:
                                st.markdown("**С RAG**")
                                st.write(f"Ожидалось: `{det_rag['expected']}`")
                                st.write(f"Получено: `{det_rag['actual']}`")
                                st.success("Правильно") if det_rag['correct'] else st.error("Неправильно")
                            with col_n:
                                st.markdown("**Без RAG**")
                                st.write(f"Ожидалось: `{det_no['expected']}`")
                                st.write(f"Получено: `{det_no['actual']}`")
                                st.success("Правильно") if det_no['correct'] else st.error("Неправильно")

            except Exception as e:
                status.update(label="❌ Ошибка", state="error")
                st.error(f"Ошибка: {e}")

# ====================== TAB 2: База знаний ======================
with tab2:
    st.subheader("➕ Добавить новую запись в базу знаний")

    with st.form("add_context_form", clear_on_submit=True):
        task_desc = st.text_area("Описание задачи", height=120, 
                                placeholder="Напиши функцию, которая ищет максимальный элемент массива...")
        
        solution_code = st.text_area("Код решения", height=250,
                                    placeholder="def solve(arr):\n    return max(arr)")
        
        submitted = st.form_submit_button("💾 Добавить в базу знаний", type="primary")
        
        if submitted:
            if not task_desc or not solution_code:
                st.error("Оба поля обязательны!")
            else:
                success, message = save_to_context(task_desc, solution_code)
                if success:
                    st.success(message)
                    st.balloons()
                else:
                    st.warning(message)

    st.divider()
    st.subheader("Текущая база знаний")

    try:
        with open("data/context.json", "r", encoding="utf-8") as f:
            context_data = json.load(f)
        
        st.info(f"Всего задач в базе: **{len(context_data)}**")
        
        for i, item in enumerate(context_data, 1):
            with st.expander(f"{i}. {item.get('task', 'Без названия')[:80]}..."):
                st.markdown("**Задача:**")
                st.write(item.get('task', '—'))
                st.markdown("**Решение:**")
                st.code(item.get('solution', '—'), language="python")
    except Exception as e:
        st.error(f"Не удалось загрузить базу знаний: {e}")

st.caption("Курсовая работа • RAG-system • 2026")