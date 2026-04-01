import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import sys
import mlflow
import os

# 🔹 Абсолютный путь к хранилищу артефактов (общий для всех задач)
MLFLOW_TRACKING_DIR = "/var/lib/jenkins/mlruns"
os.makedirs(MLFLOW_TRACKING_DIR, exist_ok=True)

mlflow.set_tracking_uri(f"file://{MLFLOW_TRACKING_DIR}")
mlflow.set_experiment("income_prediction")

def train_model(df):
    """Обучение Logistic Regression + GridSearchCV + MLflow"""
    
    # 1. Очистка данных от NaN
    df = df.dropna()
    
    X = df.drop('income', axis=1)
    y = df['income']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 2. Параметры для GridSearchCV
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

    # 3. Обучение
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # 4. Предсказания и метрики
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    # 5. Вывод метрик в STDERR (чтобы не попасть в best_model.txt)
    print(f"\n=== Метрики модели на тесте ===", file=sys.stderr)
    print(f"Accuracy:  {accuracy:.4f}", file=sys.stderr)
    print(f"Precision: {precision:.4f}", file=sys.stderr)
    print(f"Recall:    {recall:.4f}", file=sys.stderr)
    print(f"F1-score:  {f1:.4f}", file=sys.stderr)
    print(f"ROC-AUC:   {roc_auc:.4f}", file=sys.stderr)
    print(f"Лучшие параметры: {grid_search.best_params_}", file=sys.stderr)

    # 6. MLflow
    mlflow.set_experiment("income_prediction")
    
    model_path = ""
    
    with mlflow.start_run() as run:
        # Логирование параметров и метрик
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_param("cv_folds", 3)
        mlflow.log_param("scoring_metric", "f1")
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("cv_best_f1", grid_search.best_score_)

        # ✅ Логирование модели в MLflow
        mlflow.sklearn.log_model(best_model, "model")
        
        # ✅ Получаем run_id текущего запуска
        run_id = run.info.run_id
        
        # ✅ Формируем правильный URI для mlflow models serve
        mlflow_model_uri = f"runs:/{run_id}/model"
        
        print(f"\n✅ MLflow model URI: {mlflow_model_uri}", file=sys.stderr)

    # ✅ Записываем URI в best_model.txt (это прочтёт deploy-задача)
    with open("best_model.txt", "w", encoding="utf-8") as f:
        f.write(mlflow_model_uri.strip())
    
    print(f"✅ best_model.txt записан: {mlflow_model_uri}", file=sys.stderr)

    return True

if __name__ == "__main__":
    df = pd.read_csv('processed_adult.csv')
    train_model(df)
