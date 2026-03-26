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
    
    # Удаляем все строки с NaN
    df = df.dropna()
    
    X = df.drop('income', axis=1)
    y = df['income']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Параметры для GridSearchCV
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'class_weight': ['balanced', None],
        'max_iter': [1000]
    }

    base_model = LogisticRegression(random_state=42)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )

    # Обучение + поиск лучших параметров
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Предсказания на тесте
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    # Метрики на тестовой выборке
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Вывод метрик в STDERR
    print(f"\n=== Метрики модели на тесте ===", file=sys.stderr)
    print(f"Accuracy:  {accuracy:.4f}", file=sys.stderr)
    print(f"Precision: {precision:.4f}", file=sys.stderr)
    print(f"Recall:    {recall:.4f}", file=sys.stderr)
    print(f"F1-score:  {f1:.4f}", file=sys.stderr)
    print(f"ROC-AUC:   {roc_auc:.4f}", file=sys.stderr)
    print(f"Лучшие параметры: {grid_search.best_params_}", file=sys.stderr)

    # MLflow
    mlflow.set_experiment("income_prediction")
    
    # 🔥 КРИТИЧЕСКОЕ: получаем artifact_uri ВНУТРИ контекста 🔥
    with mlflow.start_run() as run:
        # Логирование параметров
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_param("cv_folds", 3)
        mlflow.log_param("scoring_metric", "f1")
        
        # Логирование метрик
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("cv_best_f1", grid_search.best_score_)

        # Сохранение модели
        mlflow.sklearn.log_model(best_model, "model")
        joblib.dump(best_model, "model_income.pkl")
        joblib.dump(X.columns.tolist(), "feature_columns.pkl")
        joblib.dump(grid_search.best_params_, "best_params.pkl")
        
        # 🔥 ПОЛУЧАЕМ artifact_uri ВНУТРИ контекста — это гарантирует правильный run_id с m- 🔥
        artifact_uri = mlflow.get_artifact_uri("model")
    
    # 🔥 Конвертируем URI в локальный путь 🔥
    # artifact_uri имеет вид: file:///.../mlruns/1/m-<run_id>/artifacts/model
    if artifact_uri.startswith("file://"):
        model_path = artifact_uri[7:]  # Убираем "file://"
    else:
        model_path = artifact_uri
    
    # Записываем ТОЛЬКО путь в best_model.txt
    with open("best_model.txt", "w") as f:
        f.write(model_path.strip())
    
    print(f"\n✅ Модель сохранена. Путь: {model_path}", file=sys.stderr)

    return True


if __name__ == "__main__":
    df = pd.read_csv('processed_adult.csv')
    train_model(df)
