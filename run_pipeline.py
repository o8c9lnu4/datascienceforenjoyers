#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Главный скрипт-менеджер для управления всем проектом
Запускает все этапы обработки данных в правильном порядке
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

# Настройка кодировки для Windows
if sys.platform == 'win32':
    try:
        # Пытаемся установить UTF-8 для вывода
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

# ============================================
# НАСТРОЙКИ
# ============================================
REPORTS_DIR = "reports"  # Папка для всех отчетов
DATA_DIR = "."  # Папка с данными
SCRIPTS_DIR = "blog"  # Папка со скриптами

# Порядок выполнения скриптов
PIPELINE_STEPS = [
    {
        'name': 'Предобработка данных',
        'script': 'preprocessing.py',
        'description': 'Очистка и классификация текстов',
        'input_file': 'dataset.csv',
        'output_files': ['dataset_cleaned.csv', 'dataset_removed.csv'],
        'reports': ['report_cleaned.txt', 'report_removed.txt']
    },
    {
        'name': 'EDA анализ',
        'script': 'eda_report.py',
        'description': 'Исследовательский анализ данных',
        'input_file': 'dataset.csv',
        'output_files': [],
        'reports': ['eda_report.txt']
    },
    {
        'name': 'Векторизация',
        'script': 'vectorization.py',
        'description': 'Векторизация текста для ML',
        'input_file': 'dataset_cleaned.csv',
        'output_files': [],
        'reports': [],
        'optional': True  # Опциональный шаг
    },
    {
        'name': 'Визуализация векторизованных данных',
        'script': 'visualization.py',
        'description': 'Визуализация и анализ векторизованных данных',
        'input_file': None,  # Использует данные из vectorized/
        'output_files': [],
        'reports': ['vectorization_report_tfidf.txt'],
        'optional': True,  # Опциональный шаг
        'depends_on': ['Векторизация']  # Зависит от векторизации
    }
]

# ============================================
# ФУНКЦИИ
# ============================================

def create_directories(reports_dir):
    """Создание необходимых директорий"""
    Path(reports_dir).mkdir(exist_ok=True)
    print(f"[OK] Директория для отчетов: {reports_dir}/")


def check_file_exists(filepath, required=True):
    """Проверка существования файла"""
    exists = os.path.exists(filepath)
    if required and not exists:
        print(f"[ERROR] Файл не найден: {filepath}")
        return False
    elif not exists:
        print(f"[WARN] Файл отсутствует: {filepath}")
    return exists


def run_script(script_path, step_info, reports_dir, skip_if_exists=False):
    """
    Запуск скрипта
    
    Parameters:
    -----------
    script_path : str
        Путь к скрипту
    step_info : dict
        Информация о шаге
    skip_if_exists : bool
        Пропустить, если выходные файлы уже существуют
    """
    step_name = step_info['name']
    print(f"\n{'='*60}")
    print(f"ШАГ: {step_name}")
    print(f"Описание: {step_info['description']}")
    print(f"{'='*60}")
    
    # Проверка входного файла
    input_file = step_info.get('input_file')
    if input_file:
        input_path = os.path.join(DATA_DIR, input_file)
        if not check_file_exists(input_path, required=True):
            print(f"[ERROR] Пропуск шага '{step_name}': отсутствует входной файл")
            return False
    
    # Проверка выходных файлов (если нужно пропустить)
    if skip_if_exists:
        all_outputs_exist = True
        for output_file in step_info.get('output_files', []):
            output_path = os.path.join(DATA_DIR, output_file)
            if not check_file_exists(output_path, required=False):
                all_outputs_exist = False
                break
        
        if all_outputs_exist and step_info.get('output_files'):
            print(f"[SKIP] Пропуск шага '{step_name}': выходные файлы уже существуют")
            return True
    
    # Проверка существования скрипта
    if not os.path.exists(script_path):
        print(f"[ERROR] Скрипт не найден: {script_path}")
        return False
    
    print(f"[RUN] Запуск: {script_path}")
    print(f"[...] Выполнение...")
    
    try:
        # Определение рабочей директории (корень проекта, где находится run_pipeline.py)
        # Скрипты должны запускаться из корня проекта, чтобы правильно находить файлы
        project_root = os.path.dirname(os.path.abspath(__file__)) or os.getcwd()
        
        # Запуск скрипта из корня проекта
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=project_root,
            capture_output=False,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            print(f"[OK] Шаг '{step_name}' выполнен успешно!")
            
            # Проверка выходных файлов
            for output_file in step_info.get('output_files', []):
                output_path = os.path.join(DATA_DIR, output_file)
                if check_file_exists(output_path, required=False):
                    file_size = os.path.getsize(output_path) / (1024 * 1024)
                    print(f"   [FILE] {output_file} ({file_size:.2f} MB)")
            
            # Проверка отчетов (они уже должны быть в reports/, но проверим)
            for report_file in step_info.get('reports', []):
                # Проверяем в корне (старое место)
                report_source = os.path.join(DATA_DIR, report_file)
                # Проверяем в reports (новое место)
                report_in_reports = os.path.join(reports_dir, report_file)
                
                if os.path.exists(report_source):
                    # Перемещаем из корня в reports
                    try:
                        import shutil
                        # Если файл уже существует в reports, создаем версию с timestamp
                        if os.path.exists(report_in_reports):
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            name, ext = os.path.splitext(report_file)
                            report_dest = os.path.join(reports_dir, f"{name}_{timestamp}{ext}")
                        else:
                            report_dest = report_in_reports
                        
                        shutil.move(report_source, report_dest)
                        print(f"   [REPORT] Отчет перемещен: {os.path.basename(report_dest)}")
                    except Exception as e:
                        print(f"   [WARN] Не удалось переместить отчет: {e}")
                elif os.path.exists(report_in_reports):
                    print(f"   [REPORT] Отчет уже в reports/: {report_file}")
            
            return True
        else:
            print(f"[ERROR] Шаг '{step_name}' завершился с ошибкой (код: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"[ERROR] ОШИБКА при выполнении шага '{step_name}': {e}")
        import traceback
        traceback.print_exc()
        return False


def run_pipeline(steps_to_run=None, skip_existing=False, skip_optional=False, reports_dir=None):
    """
    Запуск всего пайплайна
    
    Parameters:
    -----------
    steps_to_run : list, optional
        Список индексов шагов для выполнения (None = все)
    skip_existing : bool
        Пропускать шаги, если выходные файлы уже существуют
    skip_optional : bool
        Пропускать опциональные шаги
    reports_dir : str, optional
        Директория для отчетов (по умолчанию: REPORTS_DIR)
    """
    if reports_dir is None:
        reports_dir = REPORTS_DIR
    
    print("="*60)
    print("ЗАПУСК ПАЙПЛАЙНА ОБРАБОТКИ ДАННЫХ")
    print("="*60)
    print(f"Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Создание директорий
    create_directories(reports_dir)
    
    # Определение шагов для выполнения
    if steps_to_run is None:
        steps_to_run = list(range(len(PIPELINE_STEPS)))
    
    # Фильтрация опциональных шагов
    if skip_optional:
        steps_to_run = [i for i in steps_to_run if not PIPELINE_STEPS[i].get('optional', False)]
    
    print(f"\n📋 Будет выполнено шагов: {len(steps_to_run)}")
    for i in steps_to_run:
        step = PIPELINE_STEPS[i]
        optional_mark = " (опционально)" if step.get('optional', False) else ""
        print(f"   {i+1}. {step['name']}{optional_mark}")
    
    # Запуск шагов
    success_count = 0
    failed_steps = []
    
    for i, step_info in enumerate(PIPELINE_STEPS):
        if i not in steps_to_run:
            continue
        
        # Пропуск опциональных шагов, если указано
        if skip_optional and step_info.get('optional', False):
            print(f"\n[SKIP] Пропуск опционального шага: {step_info['name']}")
            continue
        
        script_path = os.path.join(SCRIPTS_DIR, step_info['script'])
        success = run_script(script_path, step_info, reports_dir, skip_if_exists=skip_existing)
        
        if success:
            success_count += 1
        else:
            failed_steps.append(step_info['name'])
            # Можно продолжить или остановиться
            print(f"[WARN] Продолжение выполнения пайплайна...")
    
    # Итоговый отчет
    print("\n" + "="*60)
    print("ИТОГОВЫЙ ОТЧЕТ")
    print("="*60)
    print(f"[OK] Успешно выполнено: {success_count}/{len(steps_to_run)} шагов")
    
    if failed_steps:
        print(f"[ERROR] Ошибки в шагах: {', '.join(failed_steps)}")
    else:
        print("[SUCCESS] Все шаги выполнены успешно!")
    
    print(f"\n[DIR] Отчеты сохранены в: {reports_dir}/")
    print(f"[TIME] Время завершения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    return len(failed_steps) == 0


def list_steps():
    """Вывод списка доступных шагов"""
    print("="*60)
    print("ДОСТУПНЫЕ ШАГИ ПАЙПЛАЙНА")
    print("="*60)
    for i, step in enumerate(PIPELINE_STEPS):
        optional_mark = " (опционально)" if step.get('optional', False) else ""
        print(f"\n{i+1}. {step['name']}{optional_mark}")
        print(f"   Описание: {step['description']}")
        print(f"   Скрипт: {step['script']}")
        if step.get('input_file'):
            print(f"   Входной файл: {step['input_file']}")
        if step.get('output_files'):
            print(f"   Выходные файлы: {', '.join(step['output_files'])}")
        if step.get('reports'):
            print(f"   Отчеты: {', '.join(step['reports'])}")


# ============================================
# ОСНОВНАЯ ПРОГРАММА
# ============================================

def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(
        description='Главный скрипт-менеджер для управления проектом обработки данных',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python run_pipeline.py                    # Запустить весь пайплайн
  python run_pipeline.py --steps 1 2         # Запустить только шаги 1 и 2
  python run_pipeline.py --skip-existing    # Пропустить существующие файлы
  python run_pipeline.py --list              # Показать список шагов
  python run_pipeline.py --skip-optional     # Пропустить опциональные шаги
        """
    )
    
    parser.add_argument('--steps', '-s',
                       type=int,
                       nargs='+',
                       help='Номера шагов для выполнения (например: 1 2 3)')
    
    parser.add_argument('--skip-existing', '-e',
                       action='store_true',
                       help='Пропустить шаги, если выходные файлы уже существуют')
    
    parser.add_argument('--skip-optional', '-o',
                       action='store_true',
                       help='Пропустить опциональные шаги')
    
    parser.add_argument('--list', '-l',
                       action='store_true',
                       help='Показать список доступных шагов')
    
    parser.add_argument('--reports-dir', '-r',
                       default=REPORTS_DIR,
                       help=f'Директория для отчетов (по умолчанию: {REPORTS_DIR})')
    
    args = parser.parse_args()
    
    if args.list:
        list_steps()
        return
    
    # Определение шагов для выполнения
    steps_to_run = args.steps
    if steps_to_run:
        # Преобразование в 0-based индексы
        steps_to_run = [s - 1 for s in steps_to_run if 1 <= s <= len(PIPELINE_STEPS)]
        if not steps_to_run:
            print("[ERROR] Некорректные номера шагов")
            return
    
    # Запуск пайплайна
    success = run_pipeline(
        steps_to_run=steps_to_run,
        skip_existing=args.skip_existing,
        skip_optional=args.skip_optional,
        reports_dir=args.reports_dir
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

