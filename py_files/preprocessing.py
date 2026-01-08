#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Предобработка данных
Очистка и классификация текстов
"""

import pandas as pd
import re
import os
from collections import Counter
import sys

# Попытка импортировать tqdm, если нет - используем заглушку
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Заглушка для tqdm
    class tqdm:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def update(self, n=1):
            pass
        @staticmethod
        def pandas(*args, **kwargs):
            pass

# Упрощённая версия без NLTK для быстрой обработки

# ----------------------------
# 1. Настройки
# ----------------------------
INPUT_FILE = "dataset.csv"

# Выходные файлы
CLEANED_OUTPUT = "dataset_cleaned.csv"
REMOVED_OUTPUT = "dataset_removed.csv"

# Папка для отчетов
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)
REPORT_CLEANED = os.path.join(REPORTS_DIR, "report_cleaned.txt")
REPORT_REMOVED = os.path.join(REPORTS_DIR, "report_removed.txt")

# ----------------------------
# Ключевые слова: доставка (расширено)
# ----------------------------
DELIVERY_KEYWORDS = {
    'посылк', 'доставк', 'трек', 'отправлени', 'получ', 'забрать', 'пункт', 'выдач',
    'статус', 'задерж', 'потеря', 'поврежд', 'возврат', 'отправ', 'адрес', 'курьер',
    'почта', '5post', 'пятерочк', 'перекресток', 'магазин', 'ящик', 'локер',
    'проблем', 'жалоб', 'ошибк', 'измени', 'отмен', 'заказ', 'товар', 'брак',
    'ждать', 'придет', 'придёт', 'сколько', 'где', 'почему', 'когда', 'отследить',
    # --- ДОБАВЛЕНО для постаматов ---
    'постамат', 'автомат', 'терминал', 'ячейк', 'локер', 'откры', 'пуст', 'инвентаризац', 'срок хранения'
}

# Защита частых жалоб
PROTECTED_PHRASES = {
    'не работает постамат',
    'не открылась ячейка',
    'не открывается ячейка',
    'продлить срок хранения',
    'не работает автомат',
    'пустая ячейка',
    'инвентаризация постамата',
    'постамат не работает',
    'ячейка не открылась',
    'не работает терминал',
    'автомат не работает',
    'ячейка пустая'
}

NON_INFORMATIVE_PHRASES = {"спасибо", "ок", "да", "нет", "большое", "большое спасибо"}

OPERATOR_KEYWORDS = {
    'оператор', 'менеджер', 'консультант', 'помощь', 'поддержка', 
    'связаться', 'позвонить', 'перевести', 'соединить',
    'специалист', 'человек', 'работник', 'сотрудник', 'консультация', 
    'местонахождение', 'техподдержка', 'местонахождени', 'техподдержк'
}


def classify_relevance(text):
    if not isinstance(text, str) or not text.strip():
        return False, "empty_or_not_string"
    
    text_clean = text.strip().lower()
    
    # 🔒 Защита известных релевантных фраз
    if text_clean in PROTECTED_PHRASES:
        return True, "protected_phrase"
    
    if len(text_clean) <= 2:
        return False, "too_short"
    
    words = text_clean.split()

    if text_clean in NON_INFORMATIVE_PHRASES:
        return False, "exact_greeting"
    
    if all(word in NON_INFORMATIVE_PHRASES for word in words):
        return False, "all_non_informative"
    
    if len(words) <= 2 and all(word in NON_INFORMATIVE_PHRASES for word in words):
        return False, "short_non_informative"

    # Проверка ключевых слов
    has_delivery = any(kw in word for word in words for kw in DELIVERY_KEYWORDS)
    has_operator = any(kw in text_clean for kw in OPERATOR_KEYWORDS)

    if has_delivery and has_operator:
        return True, "delivery_and_operator"
    elif has_delivery:
        return True, "delivery_related"
    elif has_operator:
        return True, "operator_request"
    
    if len(words) > 4:
        return True, "long_phrase"
    
    return False, "no_relevant_content"


def main():
    """Основная функция"""
    # ----------------------------
    # 2. Загрузка данных (с батч-обработкой для больших файлов)
    # ----------------------------
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Файл {INPUT_FILE} не найден в: {os.getcwd()}")

    # Проверяем размер файла для определения стратегии обработки
    file_size_mb = os.path.getsize(INPUT_FILE) / (1024 * 1024)
    print(f"Размер файла: {file_size_mb:.1f} MB")

    # Для файлов больше 500MB используем батч-обработку
    USE_BATCH = file_size_mb > 500
    BATCH_SIZE = 50000  # строк на батч

    if USE_BATCH:
        print(f"Используется батч-обработка (батч = {BATCH_SIZE} строк)")
        # Сначала загружаем небольшую выборку для определения структуры
        df_sample = pd.read_csv(INPUT_FILE, nrows=100)
        text_col = 'text' if 'text' in df_sample.columns else df_sample.columns[1] if len(df_sample.columns) > 1 else df_sample.columns[0]
        print(f"Определён текстовый столбец: '{text_col}'")
        print(f"Начинаем батч-обработку...")
    else:
        print("Загрузка всех данных в память...")
        df = pd.read_csv(INPUT_FILE)
        print(f"Загружено {len(df)} строк. Столбцы: {list(df.columns)}")
        text_col = 'text' if 'text' in df.columns else df.columns[1] if len(df.columns) > 1 else df.columns[0]
        if text_col not in df.columns:
            raise ValueError(f"Ожидаемый столбец '{text_col}' отсутствует.")

    # ----------------------------
    # 3. Очистка текста
    # ----------------------------
    def basic_clean(text):
        if not isinstance(text, str):
            return ""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    if USE_BATCH:
        # Батч-обработка для больших файлов
        all_cleaned = []
        all_removed = []
        total_rows = sum(1 for _ in open(INPUT_FILE)) - 1  # -1 для заголовка
        
        with tqdm(total=total_rows, desc="Обработка") as pbar:
            for chunk in pd.read_csv(INPUT_FILE, chunksize=BATCH_SIZE):
                # Очистка
                chunk['cleaned'] = chunk[text_col].apply(basic_clean)
                chunk['cleaned_for_classification'] = chunk['cleaned']
                
                # Классификация
                results = chunk['cleaned_for_classification'].apply(lambda x: classify_relevance(x))
                chunk['is_relevant'] = results.apply(lambda x: x[0])
                chunk['removal_reason'] = results.apply(lambda x: x[1])
                
                # Разделение
                df_cleaned_chunk = chunk[chunk['is_relevant']].copy()
                df_removed_chunk = chunk[~chunk['is_relevant']].copy()
                
                all_cleaned.append(df_cleaned_chunk)
                all_removed.append(df_removed_chunk)
                
                pbar.update(len(chunk))
        
        # Объединение результатов
        print("Объединение результатов...")
        df_cleaned = pd.concat(all_cleaned, ignore_index=True)
        df_removed = pd.concat(all_removed, ignore_index=True)
        
        print(f"✅ Оставлено: {len(df_cleaned)}")
        print(f"🗑️ Удалено: {len(df_removed)}")
        
        # Удаление дубликатов
        print("Удаление дубликатов...")
        before_dedup = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates(subset=['cleaned'])
        after_dedup = len(df_cleaned)
        print(f"Удалено дубликатов: {before_dedup - after_dedup}")
        
        # Сохранение
        print("Сохранение результатов...")
        df_cleaned.to_csv(CLEANED_OUTPUT, index=False, encoding='utf-8')
        df_removed.to_csv(REMOVED_OUTPUT, index=False, encoding='utf-8')
        
    else:
        # Обычная обработка для небольших файлов
        # Используем только базовую очистку (быстрее в разы!)
        print("Очистка текста...")
        if HAS_TQDM:
            tqdm.pandas(desc="Очистка")
            df['cleaned'] = df[text_col].progress_apply(basic_clean)
        else:
            df['cleaned'] = df[text_col].apply(basic_clean)
        df['cleaned_for_classification'] = df['cleaned']

        # ----------------------------
        # 4. Применение классификации (с прогресс-баром)
        # ----------------------------
        print("Классификация запросов...")
        if HAS_TQDM:
            tqdm.pandas(desc="Классификация")
            results = df['cleaned_for_classification'].progress_apply(lambda x: classify_relevance(x))
        else:
            results = df['cleaned_for_classification'].apply(lambda x: classify_relevance(x))
        df['is_relevant'] = results.apply(lambda x: x[0])
        df['removal_reason'] = results.apply(lambda x: x[1])

        # Разделение
        df_cleaned = df[df['is_relevant']].copy()
        df_removed = df[~df['is_relevant']].copy()

        print(f"✅ Оставлено: {len(df_cleaned)}")
        print(f"🗑️ Удалено: {len(df_removed)}")

        # ----------------------------
        # 5. Удаление дубликатов (только в релевантных)
        # ----------------------------
        print("Удаление дубликатов...")
        before_dedup = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates(subset=['cleaned'])  # Используем базовую очистку
        after_dedup = len(df_cleaned)
        print(f"Удалено дубликатов: {before_dedup - after_dedup}")

        # ----------------------------
        # 6. Сохранение
        # ----------------------------
        df_cleaned.to_csv(CLEANED_OUTPUT, index=False, encoding='utf-8')
        df_removed.to_csv(REMOVED_OUTPUT, index=False, encoding='utf-8')

    # ----------------------------
    # 7. Отчёты
    # ----------------------------
    # Отчёт: очищенные
    report_clean = [
        "=== ОТЧЁТ: РЕЛЕВАНТНЫЕ ЗАПРОСЫ ===\n",
        f"Всего релевантных: {len(df_cleaned)}",
        f"После дедупликации: {after_dedup}\n",
        "Метки отсутствуют — требуется разметка для обучения модели.\n",
        "УПРОЩЁННАЯ ОБРАБОТКА:",
        "✓ Базовая очистка текста (быстро)",
        "✓ Удаление знаков препинания",
        "✓ Нормализация пробелов",
        "✓ Колонка 'cleaned' содержит обработанный текст"
    ]

    # Отчёт: удалённые
    reason_counts = Counter(df_removed['removal_reason'])
    top_phrases = Counter(df_removed[text_col].fillna("").astype(str)).most_common(20)

    report_removed = ["=== ОТЧЁТ: УДАЛЁННЫЕ ЗАПРОСЫ ===\n", f"Всего удалено: {len(df_removed)}\n"]
    report_removed.append("Причины удаления:")
    for reason, cnt in reason_counts.most_common():
        report_removed.append(f"  {reason}: {cnt}")

    report_removed.append("\nТоп-20 удалённых фраз:")
    for phrase, cnt in top_phrases:
        if phrase.strip():
            report_removed.append(f"  '{phrase}' — {cnt}")

    # Сохранение отчётов
    with open(REPORT_CLEANED, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_clean))

    with open(REPORT_REMOVED, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_removed))

    print(f"\n✅ Готово!")
    print(f"Релевантные → {CLEANED_OUTPUT}")
    print(f"Удалённые → {REMOVED_OUTPUT}")
    print(f"Отчёты → {REPORT_CLEANED}, {REPORT_REMOVED}")


if __name__ == "__main__":
    main()



