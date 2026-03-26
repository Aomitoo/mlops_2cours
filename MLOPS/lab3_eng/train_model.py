import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import joblib
import sys
import os


def train_model(df):
    """Обучение Logistic Regression + GridSearchCV + MLflow"""
    
    # 🔥 Критическое исправление: удаляем ВСЕ строки с любыми NaN 🔥
    df = df.dropna()
    
    X = df.drop('income', axis=1)
    y = df['income']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 🔥 Параметры для GridSearchCV (подберите под вашу задачу) 🔥
    param_grid = {
        'C': [0.01, 0.1, 1, 10],                    # Обратная сила регуляризации
        'penalty': ['l1', 'l2'],                     # Тип регуляризации
        'solver': ['liblinear', 'saga'],             # Оптимизатор (поддерживают l1 и l2)
        'class_weight': ['balanced', None],          # Баланс классов
        'max_iter': [1000]                           # Максимум итераций
    }

    # 🔥 Базовая модель 🔥
    base_model = LogisticRegression(random_state=42)

    # 🔥 GridSearchCV: перебор параметров с кросс-валидацией 🔥
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,                    # 3-fold кросс-валидация
        scoring='f1',           # Метрика для выбора лучшей модели
        n_jobs=-1,              # Использовать все ядра
        verbose=1               # Вывод прогресса в stderr
    )

    # 🔥 Вывод метрик в STDERR (не попадёт в best_model.txt) 🔥
    print(f"\n🔍 Запуск GridSearchCV с {len(param_grid['C']) * len(param_grid['penalty']) * len(param_grid['solver']) * len(param_grid['class_weight'])} комбинациями параметров...", file=sys.stderr)

    # Обучение + поиск лучших параметров
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # 🔥 Вывод лучших параметров в stderr 🔥
    print(f"\n✅ Лучшие параметры: {grid_search.best_params_}", file=sys.stderr)
    print(f"✅ Лучший F1 (CV): {grid_search.best_score_:.4f}", file=sys.stderr)

    # Предсказания на тесте
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    # Метрики на тестовой выборке
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    # 🔥 Вывод финальных метрик в STDERR 🔥
    print(f"\n=== Метрики модели на тесте ===", file=sys.stderr)
    print(f"Accuracy:  {accuracy:.4f}", file=sys.stderr)
    print(f"Precision: {precision:.4f}", file=sys.stderr)
    print(f"Recall:    {recall:.4f}", file=sys.stderr)
    print(f"F1-score:  {f1:.4f}", file=sys.stderr)
    print(f"ROC-AUC:   {roc_auc:.4f}", file=sys.stderr)

    # MLflow
    mlflow.set_experiment("income_prediction")
    
    # 🔥 Получаем run_id ТЕКУЩЕГО запуска внутри контекста 🔥
    with mlflow.start_run() as run:
        current_run_id = run.info.run_id
        experiment_id = mlflow.get_experiment_by_name("income_prediction").experiment_id
        
        # 🔥 Логирование лучших параметров из GridSearch 🔥
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_param("cv_folds", 3)
        mlflow.log_param("scoring_metric", "f1")
        mlflow.log_param("test_size", 0.3)
        mlflow.log_param("random_state", 42)

        # 🔥 Логирование метрик на тесте 🔥
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("cv_best_f1", grid_search.best_score_)

        # 🔥 Сохранение лучшей модели 🔥
        mlflow.sklearn.log_model(best_model, "model")
        joblib.dump(best_model, "model_income.pkl")
        joblib.dump(X.columns.tolist(), "feature_columns.pkl")
        joblib.dump(grid_search.best_params_, "best_params.pkl")  # Сохраняем лучшие параметры
    
    # 🔥 Конструируем путь к модели ТЕКУЩЕГО запуска (без search_runs!) 🔥
    workspace_dir = os.getcwd()
    model_path = os.path.join(
        workspace_dir,
        "mlruns",
        str(experiment_id),
        current_run_id,
        "artifacts",
        "model"
    )
    
    # Записываем ТОЛЬКО путь в best_model.txt (без метрик!)
    with open("best_model.txt", "w") as f:
        f.write(model_path.strip())
    
    # Информируем в stderr
    print(f"\n✅ Модель сохранена. Путь: {model_path}", file=sys.stderr)
    print(f"✅ Run ID: {current_run_id}", file=sys.stderr)
    print(f"✅ best_model.txt записан", file=sys.stderr)

    return True


# === Основной запуск ===
if __name__ == "__main__":
    df = pd.read_csv('processed_adult.csv')
    train_model(df)
