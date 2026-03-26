import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import joblib
import sys
import os


def train_model(df):
    """Обучение Logistic Regression + MLflow"""
    
    # 🔥 Критическое исправление: удаляем ВСЕ строки с любыми NaN 🔥
    df = df.dropna()
    
    X = df.drop('income', axis=1)
    y = df['income']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 🔥 Исправлено: убран пробел в имени класса 🔥
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # 🔥 Исправлено: убран пробел в переменной 🔥
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Метрики
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    # 🔥 Исправлено: убран пробел в переменной 🔥
    roc_auc = roc_auc_score(y_test, y_proba)

    # 🔥 Вывод метрик в STDERR (не попадёт в best_model.txt) 🔥
    print(f"\n=== Метрики модели ===", file=sys.stderr)
    print(f"Accuracy:  {accuracy:.4f}", file=sys.stderr)
    print(f"Precision: {precision:.4f}", file=sys.stderr)
    print(f"Recall:    {recall:.4f}", file=sys.stderr)
    print(f"F1-score:  {f1:.4f}", file=sys.stderr)
    print(f"ROC-AUC:   {roc_auc:.4f}", file=sys.stderr)

    # MLflow
    # 🔥 Исправлено: убран пробел в названии эксперимента 🔥
    mlflow.set_experiment("income_prediction")
    
    # 🔥 Получаем run_id ТЕКУЩЕГО запуска внутри контекста 🔥
    with mlflow.start_run() as run:
        current_run_id = run.info.run_id
        experiment_id = mlflow.get_experiment_by_name("income_prediction").experiment_id
        
        # 🔥 Исправлено: убраны пробелы во всех ключах параметров 🔥
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("test_size", 0.3)
        mlflow.log_param("random_state", 42)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        # 🔥 Сохранение модели — Исправлено: убран пробел в ключе "model" 🔥
        mlflow.sklearn.log_model(model, "model")
        joblib.dump(model, "model_income.pkl")
        joblib.dump(X.columns.tolist(), "feature_columns.pkl")
    
    # 🔥 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: формируем путь ПОСЛЕ завершения контекста 🔥
    # Конструируем путь вручную на основе ТЕКУЩЕГО run_id и experiment_id
    # Это гарантирует, что путь совпадёт с реальной структурой на диске
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
