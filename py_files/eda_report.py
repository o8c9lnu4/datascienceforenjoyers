#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EDA (Exploratory Data Analysis) отчёт для dataset.csv
Автоматическое определение категорий из данных
"""

import pandas as pd
import os
from collections import Counter
import re
from datetime import datetime
import sys

# Попытка импортировать sklearn (опционально)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    import numpy as np
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Предупреждение: sklearn не установлен. Некоторые функции анализа будут недоступны.")
    print("Установите: pip install scikit-learn numpy")

# Настройки
INPUT_FILE = "dataset.csv"

# Папка для отчетов
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)
OUTPUT_REPORT = os.path.join(REPORTS_DIR, "eda_report.txt")

print("Загрузка данных для EDA...")

try:
    df_sample = pd.read_csv(INPUT_FILE, nrows=10000)
    print(f"Загружена выборка: {len(df_sample)} строк")
    print(f"Столбцы: {list(df_sample.columns)}")
    
    total_rows = sum(1 for _ in open(INPUT_FILE)) - 1  # -1 для заголовка
    print(f"Всего строк в файле: {total_rows:,}")
    
except Exception as e:
    print(f"Ошибка при загрузке: {e}")
    sys.exit(1)

text_col = 'text' if 'text' in df_sample.columns else df_sample.columns[1] if len(df_sample.columns) > 1 else df_sample.columns[0]
datetime_col = 'datetime' if 'datetime' in df_sample.columns else None

print(f"\nИспользуется текстовый столбец: '{text_col}'")

report_lines = []
report_lines.append("EDA ОТЧЁТ")
report_lines.append("")

report_lines.append("="*60)
report_lines.append(" БАЗОВАЯ ИНФОРМАЦИЯ О ДАННЫХ")
report_lines.append("="*60)
report_lines.append(f"Всего строк в датасете: {total_rows:,}")
report_lines.append(f"Столбцы: {', '.join(df_sample.columns)}")
report_lines.append(f"Размер выборки для анализа: {len(df_sample):,} строк")
report_lines.append("")

report_lines.append("="*60)
report_lines.append(" АНАЛИЗ ТЕКСТОВОГО СТОЛБЦА")
report_lines.append("="*60)
null_count = df_sample[text_col].isna().sum()
report_lines.append(f"Пустых значений: {null_count} ({null_count/len(df_sample)*100:.2f}%)")
text_lengths = df_sample[text_col].fillna("").astype(str).str.len()
report_lines.append(f"\nДлина текстов:")
report_lines.append(f"  Минимум: {text_lengths.min()} символов")
report_lines.append(f"  Максимум: {text_lengths.max()} символов")
report_lines.append(f"  Среднее: {text_lengths.mean():.1f} символов")
report_lines.append(f"  Медиана: {text_lengths.median():.1f} символов")
report_lines.append(f"  Q1 (25%): {text_lengths.quantile(0.25):.1f} символов")
report_lines.append(f"  Q3 (75%): {text_lengths.quantile(0.75):.1f} символов")

word_counts = df_sample[text_col].fillna("").astype(str).str.split().str.len()
report_lines.append(f"\nКоличество слов:")
report_lines.append(f"  Минимум: {word_counts.min()} слов")
report_lines.append(f"  Максимум: {word_counts.max()} слов")
report_lines.append(f"  Среднее: {word_counts.mean():.1f} слов")
report_lines.append(f"  Медиана: {word_counts.median():.1f} слов")

report_lines.append("\n" + "="*60)
report_lines.append("ТОП-20 САМЫХ ЧАСТЫХ ФРАЗ")
report_lines.append("="*60)

top_phrases = Counter(df_sample[text_col].fillna("").astype(str).str.lower().str.strip()).most_common(20)
for i, (phrase, count) in enumerate(top_phrases, 1):
    if phrase.strip():
        report_lines.append(f"  {i:2d}. '{phrase[:60]}...' ({count} раз)")

# АВТОМАТИЧЕСКОЕ ОПРЕДЕЛЕНИЕ КАТЕГОРИЙ
print("\n[Анализ] Автоматическое определение категорий...")

# Функция для очистки и токенизации текста
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    # Удаляем знаки препинания, оставляем только буквы и цифры
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# Русские стоп-слова (частые слова, которые не несут смысловой нагрузки)
STOP_WORDS = {
    'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все',
    'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по',
    'только', 'ее', 'мне', 'было', 'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из',
    'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг', 'ли', 'если', 'уже', 'или',
    'ни', 'быть', 'был', 'него', 'до', 'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь',
    'там', 'потом', 'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть',
    'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб', 'без',
    'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет', 'ж', 'тогда', 'кто',
    'этот', 'того', 'потому', 'этого', 'какой', 'совсем', 'ним', 'здесь', 'этом',
    'один', 'почти', 'мой', 'тем', 'чтобы', 'нее', 'сейчас', 'были', 'куда', 'зачем',
    'всех', 'никогда', 'можно', 'при', 'наконец', 'два', 'об', 'другой', 'хоть',
    'после', 'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них',
    'какая', 'много', 'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо', 'свою',
    'этой', 'перед', 'иногда', 'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им',
    'более', 'всегда', 'конечно', 'всю', 'между'
}

# Очистка текстов
print("  [1/5] Очистка и подготовка текстов...")
texts_clean = df_sample[text_col].fillna("").astype(str).apply(preprocess_text)
texts_clean = texts_clean[texts_clean.str.len() > 2]  # Убираем слишком короткие

# Анализ частотности слов
print("  [2/5] Анализ частотности слов...")
all_words = []
for text in texts_clean:
    words = text.split()
    # Фильтруем стоп-слова и короткие слова
    words = [w for w in words if len(w) > 2 and w not in STOP_WORDS]
    all_words.extend(words)

word_freq = Counter(all_words)

# Анализ биграмм (двухсловных фраз)
print("  [3/5] Анализ фраз (биграммы и триграммы)...")
bigrams = []
trigrams = []
for text in texts_clean:
    words = [w for w in text.split() if len(w) > 2 and w not in STOP_WORDS]
    # Биграммы
    for i in range(len(words) - 1):
        bigrams.append(f"{words[i]} {words[i+1]}")
    # Триграммы
    for i in range(len(words) - 2):
        trigrams.append(f"{words[i]} {words[i+1]} {words[i+2]}")

bigram_freq = Counter(bigrams)
trigram_freq = Counter(trigrams)

# TF-IDF анализ для выделения значимых терминов
print("  [4/5] TF-IDF анализ для выделения ключевых терминов...")
if HAS_SKLEARN:
    try:
        # Используем только первые 5000 текстов для TF-IDF (для скорости)
        texts_for_tfidf = texts_clean.head(5000).tolist()
        if len(texts_for_tfidf) > 100:
            vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2), min_df=2, max_df=0.95)
            tfidf_matrix = vectorizer.fit_transform(texts_for_tfidf)
            feature_names = vectorizer.get_feature_names_out()
            
            # Получаем средние TF-IDF значения для каждого термина
            mean_tfidf = np.mean(tfidf_matrix.toarray(), axis=0)
            top_tfidf_indices = np.argsort(mean_tfidf)[-30:][::-1]  # Топ-30
            top_tfidf_terms = [(feature_names[i], mean_tfidf[i]) for i in top_tfidf_indices]
        else:
            top_tfidf_terms = []
    except Exception as e:
        print(f"    Предупреждение: TF-IDF анализ не выполнен ({e})")
        top_tfidf_terms = []
else:
    print("    Пропущено: требуется sklearn")
    top_tfidf_terms = []

# Кластеризация для выявления основных тем
print("  [5/5] Кластеризация текстов для выявления тем...")
texts_for_cluster = []  # Инициализация для использования в отчете
if HAS_SKLEARN:
    try:
        # Используем выборку для кластеризации
        texts_for_cluster = texts_clean.head(2000).tolist()
        if len(texts_for_cluster) > 50:
            # Определяем оптимальное количество кластеров (от 3 до 10)
            n_clusters = min(10, max(3, len(texts_for_cluster) // 200))
            
            cluster_vectorizer = TfidfVectorizer(max_features=50, ngram_range=(1, 2), min_df=2)
            cluster_matrix = cluster_vectorizer.fit_transform(texts_for_cluster)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(cluster_matrix)
            
            # Определяем ключевые слова для каждого кластера
            cluster_keywords = {}
            feature_names_cluster = cluster_vectorizer.get_feature_names_out()
            
            for i in range(n_clusters):
                cluster_center = kmeans.cluster_centers_[i]
                top_indices = np.argsort(cluster_center)[-5:][::-1]
                top_words = [feature_names_cluster[idx] for idx in top_indices]
                cluster_keywords[i] = {
                    'keywords': top_words,
                    'size': int(np.sum(clusters == i))
                }
        else:
            cluster_keywords = {}
    except Exception as e:
        print(f"    Предупреждение: Кластеризация не выполнена ({e})")
        cluster_keywords = {}
else:
    print("    Пропущено: требуется sklearn")
    cluster_keywords = {}

# Формирование отчёта о категориях
report_lines.append("\n" + "="*60)
report_lines.append(" АВТОМАТИЧЕСКИ ОПРЕДЕЛЁННЫЕ КАТЕГОРИИ")
report_lines.append("="*60)

# Топ значимых слов
report_lines.append("\n📊 ТОП-30 САМЫХ ЧАСТЫХ ЗНАЧИМЫХ СЛОВ:")
top_words = word_freq.most_common(30)
for i, (word, count) in enumerate(top_words, 1):
    pct = count / len(texts_clean) * 100 if len(texts_clean) > 0 else 0
    report_lines.append(f"  {i:2d}. '{word}': {count} раз ({pct:.1f}% текстов)")

# Топ биграмм
report_lines.append("\n📝 ТОП-20 САМЫХ ЧАСТЫХ ФРАЗ (2 слова):")
top_bigrams = bigram_freq.most_common(20)
for i, (phrase, count) in enumerate(top_bigrams, 1):
    pct = count / len(texts_clean) * 100 if len(texts_clean) > 0 else 0
    report_lines.append(f"  {i:2d}. '{phrase}': {count} раз ({pct:.1f}%)")

# Топ триграмм
report_lines.append("\n📝 ТОП-15 САМЫХ ЧАСТЫХ ФРАЗ (3 слова):")
top_trigrams = trigram_freq.most_common(15)
for i, (phrase, count) in enumerate(top_trigrams, 1):
    pct = count / len(texts_clean) * 100 if len(texts_clean) > 0 else 0
    report_lines.append(f"  {i:2d}. '{phrase}': {count} раз ({pct:.1f}%)")

# TF-IDF ключевые термины
if top_tfidf_terms:
    report_lines.append("\n🔑 КЛЮЧЕВЫЕ ТЕРМИНЫ (TF-IDF анализ, топ-20):")
    for i, (term, score) in enumerate(top_tfidf_terms[:20], 1):
        report_lines.append(f"  {i:2d}. '{term}': {score:.4f}")

# Кластеры (категории)
if cluster_keywords:
    report_lines.append("\n🎯 ОСНОВНЫЕ ТЕМАТИЧЕСКИЕ КАТЕГОРИИ (кластеризация):")
    sorted_clusters = sorted(cluster_keywords.items(), key=lambda x: x[1]['size'], reverse=True)
    for cluster_id, info in sorted_clusters:
        keywords_str = ", ".join(info['keywords'])
        pct = info['size'] / len(texts_for_cluster) * 100 if len(texts_for_cluster) > 0 else 0
        report_lines.append(f"  Категория {cluster_id+1}: {info['size']} текстов ({pct:.1f}%)")
        report_lines.append(f"    Ключевые слова: {keywords_str}")

# Автоматически определённые ключевые слова для дальнейшего анализа
auto_keywords = set()
# Добавляем топ-50 слов
auto_keywords.update([w for w, _ in word_freq.most_common(50)])
# Добавляем слова из топ биграмм
for phrase, _ in bigram_freq.most_common(30):
    auto_keywords.update(phrase.split())

report_lines.append(f"\n💡 Всего автоматически выделено уникальных ключевых слов: {len(auto_keywords)}")

# Анализ по автоматически определённым ключевым словам
report_lines.append("\n📈 АНАЛИЗ ПО АВТОМАТИЧЕСКИ ОПРЕДЕЛЁННЫМ КЛЮЧЕВЫМ СЛОВАМ (топ-25):")
keyword_counts_auto = {}
for keyword in list(auto_keywords)[:25]:  # Берем первые 25 для анализа
    count = df_sample[text_col].fillna("").astype(str).str.lower().str.contains(
        re.escape(keyword), na=False, regex=True
    ).sum()
    if count > 0:
        keyword_counts_auto[keyword] = count

sorted_keywords_auto = sorted(keyword_counts_auto.items(), key=lambda x: x[1], reverse=True)
for keyword, count in sorted_keywords_auto:
    pct = count / len(df_sample) * 100
    report_lines.append(f"  '{keyword}': {count} ({pct:.1f}%)")

# Сохранение отчёта
with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print(f"\n[OK] EDA отчёт сохранён: {OUTPUT_REPORT}")

# Вывод на экран
print("\n" + "\n".join(report_lines))


