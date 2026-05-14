import streamlit as st
import sys
import os
import ast
import json

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
    st.header("О проекте")
    st.info("Система сравнивает генерацию кода **с RAG** и **без RAG** на модели Gemini.")
    st.caption("Модель: gemini-3-flash-preview")

tab1, tab2 = st.tabs(["🚀 Тестирование", "📚 База знаний"])

with tab1:
    st.subheader("Описание задачи")
    user_input = st.text_area(
        "Введите описание задачи", 
        height=100,
        placeholder="Напиши функцию, которая проверяет, является ли число простым...",
        label_visibility="collapsed"
    )

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Тестовые входные данные")
        test_input_str = st.text_area(
            "Каждая строка — один тест", 
            height=160,
            placeholder="1,2,3\n[4,5,6]\nhello world\n10"
        )
    
    with col2:
        st.subheader("Ожидаемые результаты")
        expected_str = st.text_area(
            "Каждая строка — один результат", 
            height=160,
            placeholder="3\n[6]\nhello world\nFalse"
        )

    if st.button("Запустить сравнение RAG vs Без RAG", type="primary", use_container_width=True):
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
                if ',' in line:
                    try:
                        return [ast.literal_eval(x.strip()) if x.strip() else x.strip() 
                               for x in line.split(',')]
                    except:
                        return line
                return line

        inputs = [parse_test_line(line) for line in test_input_str.split('\n') if parse_test_line(line) is not None]
        expecteds = [parse_test_line(line) for line in expected_str.split('\n') if parse_test_line(line) is not None]

        if len(inputs) == 0:
            st.error("Введите тестовые данные")
            st.stop()
        if len(inputs) != len(expecteds):
            st.error(f"Количество тестов не совпадает: {len(inputs)} входных vs {len(expecteds)} результатов")
            st.stop()

        with st.spinner("Генерация решений и тестирование..."):
            result = generate_with_rag(user_input, inputs, expecteds)

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

            if st.button("💾 Сохранить задачу + решение (RAG) в базу знаний", type="secondary", use_container_width=True):
                success, message = save_to_context(user_input, result["code_rag"])
                if success:
                    st.success(message)
                else:
                    st.warning(message)

            if result.get("test_details_rag"):
                st.subheader("📋 Подробные результаты тестов")
                
                for det_rag, det_no in zip(result["test_details_rag"], result["test_details_no"]):
                    with st.expander(f"Тест {det_rag['test_num']}: {det_rag['input']}"):
                        col_r, col_n = st.columns(2)
                        
                        with col_r:
                            st.markdown("**С RAG**")
                            st.write(f"Ожидалось: `{det_rag['expected']}`")
                            st.write(f"Получено: `{det_rag['actual']}`")
                            if det_rag['correct']:
                                st.success("Правильно")
                            else:
                                st.error("Неправильно")
                        
                        with col_n:
                            st.markdown("**Без RAG**")
                            st.write(f"Ожидалось: `{det_no['expected']}`")
                            st.write(f"Получено: `{det_no['actual']}`")
                            if det_no['correct']:
                                st.success("Правильно")
                            else:
                                st.error("Неправильно")


with tab2:
    st.subheader("📚 База знаний (context.json)")
    
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