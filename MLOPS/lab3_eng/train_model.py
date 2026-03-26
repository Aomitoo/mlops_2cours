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
    
    # Удаляем все строки с NaN
    df = df.dropna()
    
    X = df.drop('income', axis=1)
    y = df['income']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Обучение — БЕЗ ПРОБЕЛОВ!
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # Предсказания — БЕЗ ПРОБЕЛОВ!
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Метрики
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Метрики в stderr
    print(f"\n=== Метрики модели ===", file=sys.stderr)
    print(f"Accuracy:  {accuracy:.4f}", file=sys.stderr)
    print(f"Precision: {precision:.4f}", file=sys.stderr)
    print(f"Recall:    {recall:.4f}", file=sys.stderr)
    print(f"F1-score:  {f1:.4f}", file=sys.stderr)
    print(f"ROC-AUC:   {roc_auc:.4f}", file=sys.stderr)

    # MLflow
    mlflow.set_experiment("income_prediction")
    with mlflow.start_run():
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

        # Сохранение модели — БЕЗ ПРОБЕЛОВ в ключах!
        mlflow.sklearn.log_model(model, "model")
        joblib.dump(model, "model_income.pkl")
        joblib.dump(X.columns.tolist(), "feature_columns.pkl")
        
        #  Записываем АБСОЛЮТНЫЙ путь в best_model.txt 
        model_uri = mlflow.get_artifact_uri("model")
        # Конвертируем file:// URI в локальный путь
        if model_uri.startswith("file://"):
            model_path = model_uri[7:]  # убираем "file://"
        else:
            model_path = model_uri
            
        with open("best_model.txt", "w") as f:
            f.write(model_path.strip())
        
        # Для отладки: выводим путь в stderr
        print(f"\nМодель сохранена. Путь: {model_path}", file=sys.stderr)

    return True

if __name__ == "__main__":
    df = pd.read_csv('processed_adult.csv')
    train_model(df)
