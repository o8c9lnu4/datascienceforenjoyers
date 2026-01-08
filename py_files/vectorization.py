#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Векторизация текста для машинного обучения
Поддерживает несколько методов: TF-IDF, Count Vectorizer, Hashing Vectorizer
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
from pathlib import Path

# Настройка кодировки для Windows (чтобы не падать на Unicode в консоли)
if sys.platform == 'win32':
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Попытка импортировать sklearn
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
    from sklearn.decomposition import TruncatedSVD
    import joblib
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("ОШИБКА: sklearn не установлен!")
    print("Установите: pip install scikit-learn")
    sys.exit(1)

# НАСТРОЙКИ
INPUT_FILE = "dataset_cleaned.csv"  # Файл с очищенными данными
TEXT_COLUMN = "cleaned"  # Столбец с текстом для векторизации

# Методы векторизации
VECTORIZATION_METHODS = {
    'tfidf': {
        'name': 'TF-IDF',
        'class': TfidfVectorizer,
        'params': {
            'max_features': 5000,  # Максимальное количество признаков
            'ngram_range': (1, 2),  # Униграммы и биграммы
            'min_df': 2,  # Минимальная частота документа
            'max_df': 0.95,  # Максимальная частота документа
            'lowercase': True,
            'analyzer': 'word'
        }
    },
    'count': {
        'name': 'Count Vectorizer (Bag of Words)',
        'class': CountVectorizer,
        'params': {
            'max_features': 5000,
            'ngram_range': (1, 2),
            'min_df': 2,
            'max_df': 0.95,
            'lowercase': True,
            'analyzer': 'word'
        }
    },
    'hash': {
        'name': 'Hashing Vectorizer',
        'class': HashingVectorizer,
        'params': {
            'n_features': 5000,  # Размерность вектора
            'ngram_range': (1, 2),
            'lowercase': True,
            'analyzer': 'word'
        }
    }
}

# ФУНКЦИИ

def load_data(file_path, text_column, sample_size=None):
    """Загрузка данных из CSV файла"""
    print(f"\n[LOAD] Загрузка данных из {file_path}...")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден!")
    
    if sample_size:
        df = pd.read_csv(file_path, nrows=sample_size)
        print(f"   Загружена выборка: {len(df):,} строк")
    else:
        df = pd.read_csv(file_path)
        print(f"   Загружено: {len(df):,} строк")
    
    if text_column not in df.columns:
        raise ValueError(f"Столбец '{text_column}' не найден! Доступные столбцы: {list(df.columns)}")
    
    # Проверка на пустые значения
    null_count = df[text_column].isna().sum()
    if null_count > 0:
        print(f"   [WARN] Найдено {null_count} пустых значений, они будут удалены")
        df = df.dropna(subset=[text_column])
    
    # Фильтрация слишком коротких текстов
    df = df[df[text_column].astype(str).str.len() > 2]
    
    print(f"   [OK] Готово к векторизации: {len(df):,} строк")
    return df


def vectorize_text(df, text_column, method='tfidf', output_dir='vectorized'):
    """
    Векторизация текста выбранным методом
    
    Parameters:
    -----------
    df : DataFrame
        Датафрейм с текстами
    text_column : str
        Название столбца с текстом
    method : str
        Метод векторизации ('tfidf', 'count', 'hash')
    output_dir : str
        Директория для сохранения результатов
    """
    if method not in VECTORIZATION_METHODS:
        raise ValueError(f"Неизвестный метод: {method}. Доступные: {list(VECTORIZATION_METHODS.keys())}")
    
    method_info = VECTORIZATION_METHODS[method]
    print(f"\n[VEC] Векторизация методом: {method_info['name']}")
    
    # Создание директории для результатов
    Path(output_dir).mkdir(exist_ok=True)
    
    # Получение текстов
    texts = df[text_column].fillna("").astype(str).tolist()
    print(f"   Обработка {len(texts):,} текстов...")
    
    # Создание векторизатора
    vectorizer_class = method_info['class']
    params = method_info['params'].copy()
    
    # Специальная обработка для HashingVectorizer
    if method == 'hash':
        vectorizer = vectorizer_class(**params)
    else:
        vectorizer = vectorizer_class(**params)
    
    # Векторизация
    print("   Выполнение векторизации...")
    vectors = vectorizer.fit_transform(texts)
    
    print(f"   [OK] Векторизация завершена!")
    print(f"   Размерность векторов: {vectors.shape}")
    print(f"   Количество признаков: {vectors.shape[1]:,}")
    
    # Сохранение векторизатора
    vectorizer_path = os.path.join(output_dir, f'vectorizer_{method}.pkl')
    joblib.dump(vectorizer, vectorizer_path)
    print(f"   [SAVE] Векторизатор сохранён: {vectorizer_path}")
    
    # Сохранение векторов в разных форматах
    base_name = f'vectors_{method}'
    
    # 1. Sparse matrix (scipy sparse format) - самый эффективный для больших данных
    sparse_path = os.path.join(output_dir, f'{base_name}_sparse.npz')
    np.savez_compressed(sparse_path, data=vectors.data, indices=vectors.indices, 
                       indptr=vectors.indptr, shape=vectors.shape)
    print(f"   [SAVE] Sparse матрица сохранена: {sparse_path}")
    
    # 2. Dense matrix (numpy array) - для небольших данных или когда нужен полный доступ
    if vectors.shape[1] <= 10000:  # Сохраняем dense только если признаков не слишком много
        dense_vectors = vectors.toarray()
        dense_path = os.path.join(output_dir, f'{base_name}_dense.npy')
        np.save(dense_path, dense_vectors)
        print(f"   [SAVE] Dense матрица сохранена: {dense_path}")
        
        # 3. CSV с метаданными (первые N признаков для просмотра)
        if vectors.shape[1] <= 1000:
            df_vectors = pd.DataFrame(dense_vectors[:, :min(100, vectors.shape[1])])
            df_vectors.columns = [f'feature_{i}' for i in df_vectors.columns]
            # Добавляем исходные данные
            df_vectors['id'] = df['id'].values if 'id' in df.columns else range(len(df_vectors))
            df_vectors['text'] = df[text_column].values
            csv_path = os.path.join(output_dir, f'{base_name}_sample.csv')
            df_vectors.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"   [SAVE] Пример векторов (CSV): {csv_path}")
    
    # 4. Создание DataFrame с метаданными и ссылкой на векторы
    df_result = df.copy()
    df_result['vector_file'] = f'{base_name}_sparse.npz'
    df_result['vector_index'] = range(len(df_result))
    metadata_path = os.path.join(output_dir, f'metadata_{method}.csv')
    df_result.to_csv(metadata_path, index=False, encoding='utf-8')
    print(f"   [SAVE] Метаданные сохранены: {metadata_path}")
    
    # Информация о признаках (для TF-IDF и Count)
    if method in ['tfidf', 'count']:
        feature_names = vectorizer.get_feature_names_out()
        feature_df = pd.DataFrame({
            'feature_index': range(len(feature_names)),
            'feature_name': feature_names
        })
        features_path = os.path.join(output_dir, f'features_{method}.csv')
        feature_df.to_csv(features_path, index=False, encoding='utf-8')
        print(f"   [SAVE] Названия признаков сохранены: {features_path}")
    
    return vectors, vectorizer


def reduce_dimensions(vectors, n_components=100, method='svd'):
    """
    Уменьшение размерности векторов
    
    Parameters:
    -----------
    vectors : sparse matrix
        Векторизованные тексты
    n_components : int
        Количество компонент для уменьшения размерности
    method : str
        Метод ('svd' - TruncatedSVD)
    """
    print(f"\n[REDUCE] Уменьшение размерности до {n_components} компонент...")
    
    if method == 'svd':
        reducer = TruncatedSVD(n_components=n_components, random_state=42)
        vectors_reduced = reducer.fit_transform(vectors)
        print(f"   [OK] Размерность уменьшена: {vectors.shape} -> {vectors_reduced.shape}")
        return vectors_reduced, reducer
    else:
        raise ValueError(f"Неизвестный метод уменьшения размерности: {method}")


def load_vectors(vector_file, metadata_file=None):
    """
    Загрузка сохранённых векторов
    
    Parameters:
    -----------
    vector_file : str
        Путь к файлу с векторами (.npz для sparse, .npy для dense)
    metadata_file : str, optional
        Путь к файлу с метаданными
    """
    print(f"\n[LOAD] Загрузка векторов из {vector_file}...")
    
    if vector_file.endswith('.npz'):
        # Sparse matrix
        loaded = np.load(vector_file, allow_pickle=True)
        from scipy.sparse import csr_matrix
        vectors = csr_matrix((loaded['data'], loaded['indices'], loaded['indptr']), 
                            shape=loaded['shape'])
        print(f"   [OK] Загружено: {vectors.shape}")
    elif vector_file.endswith('.npy'):
        # Dense matrix
        vectors = np.load(vector_file)
        print(f"   [OK] Загружено: {vectors.shape}")
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {vector_file}")
    
    if metadata_file and os.path.exists(metadata_file):
        metadata = pd.read_csv(metadata_file)
        print(f"   [OK] Метаданные загружены: {len(metadata)} строк")
        return vectors, metadata
    
    return vectors


# ОСНОВНАЯ ПРОГРАММА

def main():
    """Основная функция"""
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(
        description='Векторизация текста для машинного обучения',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python vectorization.py --method tfidf
  python vectorization.py --method count --sample 10000
  python vectorization.py --method tfidf --reduce-dim --components 200
  python vectorization.py --input dataset_cleaned.csv --method hash
        """
    )
    
    parser.add_argument('--method', '-m', 
                       choices=['tfidf', 'count', 'hash'],
                       default='tfidf',
                       help='Метод векторизации (по умолчанию: tfidf)')
    
    parser.add_argument('--input', '-i',
                       default=INPUT_FILE,
                       help=f'Входной файл с данными (по умолчанию: {INPUT_FILE})')
    
    parser.add_argument('--text-column', '-t',
                       default=TEXT_COLUMN,
                       help=f'Столбец с текстом (по умолчанию: {TEXT_COLUMN})')
    
    parser.add_argument('--sample', '-s',
                       type=int,
                       default=None,
                       help='Размер выборки (None для всех данных)')
    
    parser.add_argument('--reduce-dim', '-r',
                       action='store_true',
                       help='Уменьшить размерность векторов')
    
    parser.add_argument('--components', '-c',
                       type=int,
                       default=100,
                       help='Количество компонент при уменьшении размерности (по умолчанию: 100)')
    
    parser.add_argument('--output-dir', '-o',
                       default='vectorized',
                       help='Директория для сохранения результатов (по умолчанию: vectorized)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("ВЕКТОРИЗАЦИЯ ТЕКСТА")
    print("="*60)
    print(f"Метод: {VECTORIZATION_METHODS[args.method]['name']}")
    print(f"Входной файл: {args.input}")
    print(f"Столбец с текстом: {args.text_column}")
    if args.sample:
        print(f"Размер выборки: {args.sample:,}")
    else:
        print("Обработка всех данных")
    if args.reduce_dim:
        print(f"Уменьшение размерности: {args.components} компонент")
    print(f"Выходная директория: {args.output_dir}")
    print("="*60)
    
    # Загрузка данных
    try:
        df = load_data(args.input, args.text_column, sample_size=args.sample)
    except Exception as e:
        print(f"[ERROR] Ошибка при загрузке данных: {e}")
        sys.exit(1)
    
    # Векторизация
    try:
        vectors, vectorizer = vectorize_text(df, args.text_column, method=args.method, output_dir=args.output_dir)
        
        # Опциональное уменьшение размерности
        if args.reduce_dim and vectors.shape[1] > args.components:
            vectors_reduced, reducer = reduce_dimensions(vectors, n_components=args.components)
            
            # Сохранение уменьшенных векторов
            reduced_path = os.path.join(args.output_dir, f'vectors_{args.method}_reduced_{args.components}.npy')
            np.save(reduced_path, vectors_reduced)
            print(f"   [SAVE] Уменьшенные векторы сохранены: {reduced_path}")
            
            # Сохранение редуктора
            reducer_path = os.path.join(args.output_dir, f'reducer_{args.method}_{args.components}.pkl')
            joblib.dump(reducer, reducer_path)
            print(f"   [SAVE] Редуктор сохранён: {reducer_path}")
        
        print("\n" + "="*60)
        print("[SUCCESS] ВЕКТОРИЗАЦИЯ ЗАВЕРШЕНА УСПЕШНО!")
        print("="*60)
        print(f"\n📁 Результаты сохранены в директории: {args.output_dir}/")
        print(f"   - Векторизатор: vectorizer_{args.method}.pkl")
        print(f"   - Векторы (sparse): vectors_{args.method}_sparse.npz")
        if vectors.shape[1] <= 10000:
            print(f"   - Векторы (dense): vectors_{args.method}_dense.npy")
        print(f"   - Метаданные: metadata_{args.method}.csv")
        if args.method in ['tfidf', 'count']:
            print(f"   - Признаки: features_{args.method}.csv")
        
    except Exception as e:
        print(f"[ERROR] Ошибка при векторизации: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


