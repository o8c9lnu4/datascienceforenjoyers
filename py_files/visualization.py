#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Визуализация векторизованных данных
Создает графики и отчеты для анализа векторизованных текстов
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
from pathlib import Path
from collections import Counter

# Настройка кодировки для Windows
if sys.platform == 'win32':
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

# Попытка импортировать sklearn
try:
    from sklearn.decomposition import TruncatedSVD, PCA
    from sklearn.manifold import TSNE
    import joblib
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("ОШИБКА: sklearn не установлен!")
    print("Установите: pip install scikit-learn")
    sys.exit(1)

# Попытка импортировать библиотеки визуализации
try:
    import matplotlib
    matplotlib.use('Agg')  # Используем backend без GUI
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VIS = True
except ImportError:
    HAS_VIS = False
    print("[WARN] Предупреждение: matplotlib/seaborn не установлены!")
    print("Визуализации будут пропущены. Установите: pip install matplotlib seaborn")

# Попытка импортировать scipy для работы со sparse матрицами
try:
    from scipy.sparse import csr_matrix
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARN] Предупреждение: scipy не установлен!")
    print("Установите: pip install scipy")

# НАСТРОЙКИ
VECTORIZED_DIR = "vectorized"  # Директория с векторизованными данными
VISUALIZATIONS_DIR = "visualizations"  # Директория для сохранения графиков
REPORTS_DIR = "reports"  # Директория для отчетов

# Настройки визуализации
DEFAULT_METHOD = "tfidf"
MAX_SAMPLES_FOR_TSNE = 10000  # Максимальное количество образцов для t-SNE
MAX_SAMPLES_FOR_PLOTS = 50000  # Максимальное количество образцов для графиков

# ФУНКЦИИ

def load_vectorized_data(method='tfidf', vectorized_dir='vectorized'):
    """Загрузка векторизованных данных"""
    print(f"\n[LOAD] Загрузка векторизованных данных (метод: {method})...")
    
    # Пути к файлам
    sparse_path = os.path.join(vectorized_dir, f'vectors_{method}_sparse.npz')
    metadata_path = os.path.join(vectorized_dir, f'metadata_{method}.csv')
    features_path = os.path.join(vectorized_dir, f'features_{method}.csv')
    
    # Проверка существования файлов
    if not os.path.exists(sparse_path):
        raise FileNotFoundError(f"Файл с векторами не найден: {sparse_path}")
    
    # Загрузка sparse матрицы
    loaded = np.load(sparse_path, allow_pickle=True)
    
    # Импорт csr_matrix
    try:
        from scipy.sparse import csr_matrix
    except ImportError:
        raise ImportError("scipy необходим для загрузки sparse матриц. Установите: pip install scipy")
    
    vectors = csr_matrix((loaded['data'], loaded['indices'], loaded['indptr']), 
                        shape=loaded['shape'])
    print(f"   [OK] Векторы загружены: {vectors.shape}")
    
    # Загрузка метаданных
    metadata = None
    if os.path.exists(metadata_path):
        metadata = pd.read_csv(metadata_path)
        print(f"   [OK] Метаданные загружены: {len(metadata)} строк")
    
    # Загрузка названий признаков (для TF-IDF и Count)
    feature_names = None
    if method in ['tfidf', 'count'] and os.path.exists(features_path):
        features_df = pd.read_csv(features_path)
        feature_names = features_df['feature_name'].values
        print(f"   [OK] Названия признаков загружены: {len(feature_names)} признаков")
    
    return vectors, metadata, feature_names


def calculate_vector_statistics(vectors):
    """Расчет статистики по векторам"""
    print("\n[STATS] Расчет статистики по векторам...")
    
    stats = {}
    
    # Основные характеристики
    stats['shape'] = vectors.shape
    stats['n_samples'] = vectors.shape[0]
    stats['n_features'] = vectors.shape[1]
    
    # Разреженность
    if hasattr(vectors, 'nnz'):  # Sparse matrix
        total_elements = vectors.shape[0] * vectors.shape[1]
        stats['sparsity'] = 1 - (vectors.nnz / total_elements)
        stats['non_zero_elements'] = vectors.nnz
        stats['sparsity_percent'] = stats['sparsity'] * 100
    else:  # Dense matrix
        non_zero = np.count_nonzero(vectors)
        total_elements = vectors.size
        stats['sparsity'] = 1 - (non_zero / total_elements)
        stats['non_zero_elements'] = non_zero
        stats['sparsity_percent'] = stats['sparsity'] * 100
    
    # Статистика по строкам (документам)
    if hasattr(vectors, 'getnnz'):
        row_nnz = vectors.getnnz(axis=1)
        stats['mean_features_per_doc'] = np.mean(row_nnz)
        stats['median_features_per_doc'] = np.median(row_nnz)
        stats['min_features_per_doc'] = np.min(row_nnz)
        stats['max_features_per_doc'] = np.max(row_nnz)
    else:
        row_nnz = np.count_nonzero(vectors, axis=1)
        stats['mean_features_per_doc'] = np.mean(row_nnz)
        stats['median_features_per_doc'] = np.median(row_nnz)
        stats['min_features_per_doc'] = np.min(row_nnz)
        stats['max_features_per_doc'] = np.max(row_nnz)
    
    # Статистика по столбцам (признакам)
    if hasattr(vectors, 'getnnz'):
        col_nnz = vectors.getnnz(axis=0)
        stats['mean_docs_per_feature'] = np.mean(col_nnz)
        stats['median_docs_per_feature'] = np.median(col_nnz)
        stats['min_docs_per_feature'] = np.min(col_nnz)
        stats['max_docs_per_feature'] = np.max(col_nnz)
    else:
        col_nnz = np.count_nonzero(vectors, axis=0)
        stats['mean_docs_per_feature'] = np.mean(col_nnz)
        stats['median_docs_per_feature'] = np.median(col_nnz)
        stats['min_docs_per_feature'] = np.min(col_nnz)
        stats['max_docs_per_feature'] = np.max(col_nnz)
    
    # Значения
    if hasattr(vectors, 'data'):
        if len(vectors.data) > 0:
            stats['mean_value'] = np.mean(vectors.data)
            stats['median_value'] = np.median(vectors.data)
            stats['min_value'] = np.min(vectors.data)
            stats['max_value'] = np.max(vectors.data)
            stats['std_value'] = np.std(vectors.data)
    else:
        non_zero_values = vectors[vectors != 0]
        if len(non_zero_values) > 0:
            stats['mean_value'] = np.mean(non_zero_values)
            stats['median_value'] = np.median(non_zero_values)
            stats['min_value'] = np.min(non_zero_values)
            stats['max_value'] = np.max(non_zero_values)
            stats['std_value'] = np.std(non_zero_values)
    
    return stats


def visualize_sparsity(vectors, output_dir, method='tfidf'):
    """Визуализация разреженности матрицы"""
    if not HAS_VIS:
        return
    
    print("\n[VIZ] Создание визуализации разреженности...")
    
    if hasattr(vectors, 'getnnz'):
        row_nnz = vectors.getnnz(axis=1)
        col_nnz = vectors.getnnz(axis=0)
    else:
        row_nnz = np.count_nonzero(vectors, axis=1)
        col_nnz = np.count_nonzero(vectors, axis=0)
    
    n_samples = min(len(row_nnz), MAX_SAMPLES_FOR_PLOTS)
    sample_indices = np.random.choice(len(row_nnz), n_samples, replace=False)
    row_nnz_sample = row_nnz[sample_indices]
    
    n_features = min(len(col_nnz), MAX_SAMPLES_FOR_PLOTS)
    feature_indices = np.random.choice(len(col_nnz), n_features, replace=False)
    col_nnz_sample = col_nnz[feature_indices]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Визуализация разреженности матрицы ({method.upper()})', fontsize=16, fontweight='bold')
    
    axes[0, 0].hist(row_nnz_sample, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Количество ненулевых признаков в документе')
    axes[0, 0].set_ylabel('Частота')
    axes[0, 0].set_title('Распределение признаков по документам')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(col_nnz_sample, bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[0, 1].set_xlabel('Количество документов с признаком')
    axes[0, 1].set_ylabel('Частота')
    axes[0, 1].set_title('Распределение документов по признакам')
    axes[0, 1].grid(True, alpha=0.3)
    
    sample_size = min(100, vectors.shape[0])
    feature_size = min(100, vectors.shape[1])
    sample_idx = np.random.choice(vectors.shape[0], sample_size, replace=False)
    feature_idx = np.random.choice(vectors.shape[1], feature_size, replace=False)
    
    if hasattr(vectors, 'toarray'):
        sample_matrix = vectors[sample_idx][:, feature_idx].toarray()
    else:
        sample_matrix = vectors[np.ix_(sample_idx, feature_idx)]
    
    im = axes[1, 0].imshow(sample_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
    axes[1, 0].set_xlabel('Признаки (выборка)')
    axes[1, 0].set_ylabel('Документы (выборка)')
    axes[1, 0].set_title(f'Визуализация матрицы ({sample_size}x{feature_size})')
    plt.colorbar(im, ax=axes[1, 0])
    
    n_docs_for_box = min(100, vectors.shape[0])
    doc_indices = np.random.choice(vectors.shape[0], n_docs_for_box, replace=False)
    
    if hasattr(vectors, 'getnnz'):
        doc_features = [vectors.getrow(i).getnnz() for i in doc_indices]
    else:
        doc_features = [np.count_nonzero(vectors[i]) for i in doc_indices]
    
    axes[1, 1].boxplot(doc_features, vert=True)
    axes[1, 1].set_ylabel('Количество ненулевых признаков')
    axes[1, 1].set_title('Box plot: признаки в документах')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'sparsity_{method}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   [SAVE] Сохранено: {output_path}")


def visualize_top_features(vectors, feature_names, output_dir, method='tfidf', top_n=50):
    """Визуализация топ признаков"""
    if not HAS_VIS or feature_names is None:
        return
    
    print(f"\n[VIZ] Создание визуализации топ-{top_n} признаков...")
    
    if hasattr(vectors, 'sum'):
        feature_sums = np.array(vectors.sum(axis=0)).flatten()
    else:
        feature_sums = np.sum(vectors, axis=0)
    
    top_indices = np.argsort(feature_sums)[-top_n:][::-1]
    top_features = feature_names[top_indices]
    top_values = feature_sums[top_indices]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f'Топ-{top_n} признаков ({method.upper()})', fontsize=16, fontweight='bold')
    
    axes[0].barh(range(len(top_features)), top_values, color='steelblue')
    axes[0].set_yticks(range(len(top_features)))
    axes[0].set_yticklabels(top_features, fontsize=8)
    axes[0].set_xlabel('Сумма значений признака')
    axes[0].set_title('Топ признаков (по сумме)')
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3, axis='x')
    
    axes[1].bar(range(len(top_features)), top_values, color='coral')
    axes[1].set_xticks(range(len(top_features)))
    axes[1].set_xticklabels(top_features, rotation=45, ha='right', fontsize=8)
    axes[1].set_ylabel('Сумма значений признака (log scale)')
    axes[1].set_title('Топ признаков (логарифмическая шкала)')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'top_features_{method}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   [SAVE] Сохранено: {output_path}")


def visualize_dimension_reduction(vectors, output_dir, method='tfidf', n_samples=None, n_components=2):
    """Визуализация после уменьшения размерности (PCA/t-SNE)"""
    if not HAS_VIS:
        return
    
    print(f"\n[VIZ] Создание визуализации после уменьшения размерности...")
    
    if n_samples is None:
        n_samples = min(vectors.shape[0], MAX_SAMPLES_FOR_TSNE)
    
    if vectors.shape[0] > n_samples:
        print(f"   Используется выборка: {n_samples} из {vectors.shape[0]} документов")
        sample_indices = np.random.choice(vectors.shape[0], n_samples, replace=False)
        vectors_sample = vectors[sample_indices]
    else:
        vectors_sample = vectors
        sample_indices = np.arange(vectors.shape[0])
    
    if hasattr(vectors_sample, 'toarray'):
        print("   Преобразование sparse матрицы в dense...")
        vectors_dense = vectors_sample.toarray()
    else:
        vectors_dense = vectors_sample
    
    print("   Выполнение PCA...")
    pca = PCA(n_components=min(50, vectors_dense.shape[1]))
    vectors_pca = pca.fit_transform(vectors_dense)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    fig.suptitle(f'Уменьшение размерности ({method.upper()})', fontsize=16, fontweight='bold')
    
    axes[0, 0].scatter(vectors_pca[:, 0], vectors_pca[:, 1], alpha=0.5, s=10)
    axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% variance)')
    axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% variance)')
    axes[0, 0].set_title('PCA визуализация')
    axes[0, 0].grid(True, alpha=0.3)
    
    n_components_to_show = min(50, len(pca.explained_variance_ratio_))
    axes[0, 1].plot(range(1, n_components_to_show + 1), 
                    pca.explained_variance_ratio_[:n_components_to_show], 
                    marker='o', linewidth=2, markersize=4)
    axes[0, 1].set_xlabel('PCA компонент')
    axes[0, 1].set_ylabel('мера разброса данных')
    axes[0, 1].set_title('мера разброса данных по компонентам')
    axes[0, 1].grid(True, alpha=0.3)
    
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_[:n_components_to_show])
    axes[1, 0].plot(range(1, n_components_to_show + 1), cumulative_variance, 
                    marker='o', linewidth=2, markersize=4, color='green')
    axes[1, 0].axhline(y=0.8, color='r', linestyle='--', label='80% variance')
    axes[1, 0].axhline(y=0.9, color='orange', linestyle='--', label='90% variance')
    axes[1, 0].set_xlabel('PCA кумуляция')
    axes[1, 0].set_ylabel('общая доля главных компонентов')
    axes[1, 0].set_title('общая доля главных компонентов')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    if vectors_pca.shape[1] >= 3:
        ax_3d = fig.add_subplot(2, 2, 4, projection='3d')
        ax_3d.scatter(vectors_pca[:, 0], vectors_pca[:, 1], vectors_pca[:, 2], 
                     alpha=0.5, s=10)
        ax_3d.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
        ax_3d.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
        ax_3d.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.2f}%)')
        ax_3d.set_title('PCA 3d визуализация')
    else:
        axes[1, 1].text(0.5, 0.5, 'Недостаточно компонент\nдля 3D визуализации', 
                        ha='center', va='center', fontsize=12)
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'dimension_reduction_{method}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   [SAVE] Сохранено: {output_path}")
    
    if n_samples <= MAX_SAMPLES_FOR_TSNE:
        print("   Выполнение t-SNE (это может занять время)...")
        try:
            tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
            vectors_tsne = tsne.fit_transform(vectors_dense)
            
            plt.figure(figsize=(12, 10))
            plt.scatter(vectors_tsne[:, 0], vectors_tsne[:, 1], alpha=0.5, s=10)
            plt.xlabel('t-SNE компонента 1')
            plt.ylabel('t-SNE компонента 2')
            plt.title(f't-SNE визуализация ({method.upper()}, {n_samples} образцов)')
            plt.grid(True, alpha=0.3)
            
            output_path = os.path.join(output_dir, f'tsne_{method}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   [SAVE] Сохранено: {output_path}")
        except Exception as e:
            print(f"   [WARN] Ошибка при выполнении t-SNE: {e}")


def create_text_report(stats, output_dir, method='tfidf'):
    """Создание текстового отчета"""
    print("\n[REPORT] Создание текстового отчета...")
    
    report_lines = []
    report_lines.append("="*60)
    report_lines.append(f"ОТЧЕТ ПО ВИЗУАЛИЗАЦИИ ВЕКТОРИЗОВАННЫХ ДАННЫХ")
    report_lines.append(f"Метод: {method.upper()}")
    report_lines.append("="*60)
    report_lines.append("")
    
    report_lines.append("ОСНОВНЫЕ ХАРАКТЕРИСТИКИ:")
    report_lines.append(f"  Размерность матрицы: {stats['shape'][0]:,} x {stats['shape'][1]:,}")
    report_lines.append(f"  Количество документов: {stats['n_samples']:,}")
    report_lines.append(f"  Количество признаков: {stats['n_features']:,}")
    report_lines.append("")
    
    report_lines.append("РАЗРЕЖЕННОСТЬ:")
    report_lines.append(f"  Ненулевых элементов: {stats['non_zero_elements']:,}")
    report_lines.append(f"  Разреженность: {stats['sparsity_percent']:.2f}%")
    report_lines.append("")
    
    report_lines.append("СТАТИСТИКА ПО ДОКУМЕНТАМ:")
    report_lines.append(f"  Среднее признаков на документ: {stats['mean_features_per_doc']:.2f}")
    report_lines.append(f"  Медиана признаков на документ: {stats['median_features_per_doc']:.2f}")
    report_lines.append(f"  Мин. признаков на документ: {stats['min_features_per_doc']}")
    report_lines.append(f"  Макс. признаков на документ: {stats['max_features_per_doc']}")
    report_lines.append("")
    
    report_lines.append("СТАТИСТИКА ПО ПРИЗНАКАМ:")
    report_lines.append(f"  Среднее документов на признак: {stats['mean_docs_per_feature']:.2f}")
    report_lines.append(f"  Медиана документов на признак: {stats['median_docs_per_feature']:.2f}")
    report_lines.append(f"  Мин. документов на признак: {stats['min_docs_per_feature']}")
    report_lines.append(f"  Макс. документов на признак: {stats['max_docs_per_feature']}")
    report_lines.append("")
    
    if 'mean_value' in stats:
        report_lines.append("СТАТИСТИКА ЗНАЧЕНИЙ:")
        report_lines.append(f"  Среднее значение: {stats['mean_value']:.4f}")
        report_lines.append(f"  Медиана значений: {stats['median_value']:.4f}")
        report_lines.append(f"  Мин. значение: {stats['min_value']:.4f}")
        report_lines.append(f"  Макс. значение: {stats['max_value']:.4f}")
        report_lines.append(f"  Стд. отклонение: {stats['std_value']:.4f}")
        report_lines.append("")
    
    report_lines.append("="*60)
    
    report_path = os.path.join(output_dir, f'vectorization_report_{method}.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"   [SAVE] Сохранено: {report_path}")


# ОСНОВНАЯ ПРОГРАММА

def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(
        description='Визуализация векторизованных данных',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python visualization.py --method tfidf
  python visualization.py --method count --no-tsne
  python visualization.py --method tfidf --samples 5000
        """
    )
    
    parser.add_argument('--method', '-m',
                       choices=['tfidf', 'count', 'hash'],
                       default=DEFAULT_METHOD,
                       help=f'Метод векторизации (по умолчанию: {DEFAULT_METHOD})')
    
    parser.add_argument('--vectorized-dir', '-v',
                       default=VECTORIZED_DIR,
                       help=f'Директория с векторизованными данными (по умолчанию: {VECTORIZED_DIR})')
    
    parser.add_argument('--output-dir', '-o',
                       default=VISUALIZATIONS_DIR,
                       help=f'Директория для сохранения графиков (по умолчанию: {VISUALIZATIONS_DIR})')
    
    parser.add_argument('--samples', '-s',
                       type=int,
                       default=None,
                       help='Количество образцов для визуализации (по умолчанию: автоматически)')
    
    parser.add_argument('--no-tsne', '-t',
                       action='store_true',
                       help='Пропустить t-SNE визуализацию (может быть медленной)')
    
    parser.add_argument('--top-features', '-f',
                       type=int,
                       default=50,
                       help='Количество топ признаков для визуализации (по умолчанию: 50)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("ВИЗУАЛИЗАЦИЯ ВЕКТОРИЗОВАННЫХ ДАННЫХ")
    print("="*60)
    print(f"Метод: {args.method.upper()}")
    print(f"Директория с данными: {args.vectorized_dir}")
    print(f"Директория для графиков: {args.output_dir}")
    print("="*60)
    
    # Создание директорий
    Path(args.output_dir).mkdir(exist_ok=True)
    Path(REPORTS_DIR).mkdir(exist_ok=True)
    
    # Загрузка данных
    try:
        vectors, metadata, feature_names = load_vectorized_data(
            method=args.method,
            vectorized_dir=args.vectorized_dir
        )
    except Exception as e:
        print(f"[ERROR] Ошибка при загрузке данных: {e}")
        sys.exit(1)
    
    # Расчет статистики
    stats = calculate_vector_statistics(vectors)
    
    # Создание визуализаций
    if HAS_VIS:
        visualize_sparsity(vectors, args.output_dir, method=args.method)
        
        if feature_names is not None:
            visualize_top_features(vectors, feature_names, args.output_dir, 
                                 method=args.method, top_n=args.top_features)
        
        visualize_dimension_reduction(vectors, args.output_dir, method=args.method, 
                                    n_samples=args.samples)
    else:
        print("\n[WARN] Визуализации пропущены (библиотеки не установлены)")
    
    # Создание текстового отчета
    create_text_report(stats, REPORTS_DIR, method=args.method)
    
    print("\n" + "="*60)
    print("[SUCCESS] ВИЗУАЛИЗАЦИЯ ЗАВЕРШЕНА УСПЕШНО!")
    print("="*60)
    print(f"\n[DIR] Графики сохранены в: {args.output_dir}/")
    print(f"[DIR] Отчеты сохранены в: {REPORTS_DIR}/")


if __name__ == "__main__":
    main()





