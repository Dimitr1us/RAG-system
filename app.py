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
    user_input = st.text_area("Напишите задачу:", height=110, 
                              placeholder="Напиши функцию, которая ищет максимальный элемент в списке...")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Тестовые входные данные")
        test_input_str = st.text_area(
            "Каждая строка — один тест", 
            height=150, 
            placeholder="1,2,3\n4,5,6,7\n[10,20]\nhello"
        )
    
    with col2:
        st.subheader("Ожидаемые результаты")
        expected_str = st.text_area(
            "Каждая строка — один результат", 
            height=150,
            placeholder="3\n7\n20\nhello"
        )

    if st.button("🚀 Запустить сравнение RAG vs Без RAG", type="primary", use_container_width=True):
        if not user_input.strip():
            st.warning("Введите описание задачи")
            st.stop()

        # ==================== Новый парсинг ====================
        def parse_test_line(line: str):
            line = line.strip()
            if not line:
                return None
            try:
                # Если уже валидный Python литерал — используем как есть
                return ast.literal_eval(line)
            except:
                # Если это просто числа/строки через запятую — оборачиваем в список
                if ',' in line:
                    try:
                        # Пробуем распарсить как список
                        items = [ast.literal_eval(x.strip()) if x.strip() else x.strip() 
                                for x in line.split(',')]
                        return items
                    except:
                        return line  # если не получилось — оставляем строку
                return line  # обычная строка

        # Разбиваем по строкам
        inputs = []
        expecteds = []

        for line in test_input_str.split('\n'):
            parsed = parse_test_line(line)
            if parsed is not None:
                inputs.append(parsed)

        for line in expected_str.split('\n'):
            parsed = parse_test_line(line)
            if parsed is not None:
                expecteds.append(parsed)

        # Проверки
        if len(inputs) == 0:
            st.error("Введите хотя бы один тестовый пример")
            st.stop()

        if len(inputs) != len(expecteds):
            st.error(f"Количество строк не совпадает!\nВходных данных: {len(inputs)}, Ожидаемых результатов: {len(expecteds)}")
            st.stop()

        # ==================== Запуск ====================
        with st.spinner("Генерация и тестирование..."):
            result = generate_with_rag(user_input, inputs, expecteds)

            st.success(f"Готово за {result['time']} сек")

            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("✅ С RAG")
                st.code(result["code_rag"], language="python")
                if "accuracy_rag" in result:
                    if result.get("error_rag"):
                        st.error(result["error_rag"])
                    else:
                        st.success(f"Точность: **{result['accuracy_rag']:.1%}**")

            with col_b:
                st.subheader("⚪️ Без RAG")
                st.code(result["code_no_rag"], language="python")
                if "accuracy_no_rag" in result:
                    if result.get("error_no_rag"):
                        st.error(result["error_no_rag"])
                    else:
                        st.success(f"Точность: **{result['accuracy_no_rag']:.1%}**")

            # Подробные результаты
            if result.get("test_details_rag"):
                st.subheader("📋 Подробные результаты тестов")
                for det_rag, det_no in zip(result["test_details_rag"], result["test_details_no"]):
                    with st.expander(f"Тест {det_rag['test_num']}: Input = {det_rag['input']}"):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("**С RAG**")
                            st.write(f"Ожидалось: `{det_rag['expected']}`")
                            st.write(f"Получено: `{det_rag['actual']}`")
                            if det_rag['correct']:
                                st.success("Правильно")
                            else:
                                st.error("Неправильно")
                        with c2:
                            st.markdown("**Без RAG**")
                            st.write(f"Ожидалось: `{det_no['expected']}`")
                            st.write(f"Получено: `{det_no['actual']}`")
                            if det_no['correct']:
                                st.success("Правильно")
                            else:
                                st.error("Неправильно")

st.caption("Курсовая работа • RAG-system")