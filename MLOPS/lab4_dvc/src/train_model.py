import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import sys
import mlflow
import os
import argparse
import yaml
import json

def load_params(params_path='params.yaml'):
    with open(params_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def train_model(input_path, output_path, params_path, mlflow_dir):
    """Обучение модели с логированием в MLflow"""
    import json  
    
    params = load_params(params_path)
    
    # Настройка MLflow 
    if not os.path.isabs(mlflow_dir):
        mlflow_dir = os.path.abspath(mlflow_dir)
    os.makedirs(mlflow_dir, exist_ok=True)
    
    mlflow.set_tracking_uri(mlflow_dir)
    mlflow.set_experiment("income_prediction")
    
    # Загрузка данных
    df = pd.read_csv(input_path).dropna()
    X = df.drop('income', axis=1)
    y = df['income']
    
    # Разделение
    train_params = params['train']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=train_params['test_size'], 
        random_state=train_params['random_state'], 
        stratify=y
    )
    
    # GridSearch
    param_grid = {k: v for k, v in params['train']['grid'].items()}
    base_model = LogisticRegression(random_state=train_params['random_state'])
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=train_params['cv_folds'],
        scoring=train_params['scoring'],
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    # Оценка
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1_score': float(f1_score(y_test, y_pred)),
        'roc_auc': float(roc_auc_score(y_test, y_proba)),
        'cv_best_f1': float(grid_search.best_score_)
    }
    
    # Вывод метрик в stderr
    print(f"\n=== Метрики модели ===", file=sys.stderr)
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}", file=sys.stderr)
    print(f"Лучшие параметры: {grid_search.best_params_}", file=sys.stderr)
    
    # СОХРАНЕНИЕ МЕТРИК В JSON (для DVC)
    metrics_path = os.path.join(os.path.dirname(output_path), 'metrics.json')
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"✓ Метрики сохранены: {metrics_path}", file=sys.stderr)
    
    # MLflow logging
    with mlflow.start_run() as run:
        # Параметры
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_param("cv_folds", train_params['cv_folds'])
        mlflow.log_param("scoring_metric", train_params['scoring'])
        
        # Метрики
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        
        # Модель
        mlflow.sklearn.log_model(best_model, "model")
        run_id = run.info.run_id
        mlflow_model_uri = f"runs:/{run_id}/model"
        print(f"\n MLflow model URI: {mlflow_model_uri}", file=sys.stderr)
    
    # Сохранение URI модели
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(mlflow_model_uri.strip())
    
    print(f"✓ Модель сохранена: {mlflow_model_uri}", file=sys.stderr)
    return True  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--params', type=str, default='params.yaml')
    parser.add_argument('--mlflow-dir', type=str, default='mlruns')
    args = parser.parse_args()
    
    train_model(args.input, args.output, args.params, args.mlflow_dir)

