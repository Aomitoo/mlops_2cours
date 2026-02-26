import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import joblib
import os

def load_data(file_path='raw_adult.csv'):
    """Загрузка сырых данных"""
    df = pd.read_csv(file_path, na_values=['?', ' ?'])
    print(f"Загружено данных: {df.shape}")
    return df

def clean_data(df):
    """Очистка: пропуски, дубликаты, выбросы"""
    df = df.copy()

    # Удаляем дубликаты
    df = df.drop_duplicates()

    # Удаляем пропуски в важных колонках
    df = df.dropna(subset=['workclass', 'occupation', 'native_country'])

    # Обработка выбросов для числовых колонок
    numeric_cols = ['age', 'fnlwgt', 'education_num', 'capital_gain',
                    'capital_loss', 'hours_per_week']

    for col in numeric_cols:
        if col in df.columns:
            # Клиппинг отрицательных значений
            df[col] = df[col].clip(lower=0)

            # Замена выбросов медианой (IQR метод)
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            median = df[col].median()
            df.loc[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)), col] = median

    print(f"После очистки: {df.shape}, дубликатов: {df.duplicated().sum()}")
    return df

def feature_engineering(df):
    """Создание 5 новых признаков"""
    df = df.copy()

    # 1. Общий капитал
    df['total_capital'] = df['capital_gain'] + df['capital_loss']

    # 2. Возрастная группа
    df['age_group'] = pd.cut(df['age'], bins=[0, 25, 60, 100],
                             labels=['young', 'adult', 'senior'])

    # 3. Часов в день
    df['hours_per_day'] = df['hours_per_week'] / 5

    # 4. Образование * часы
    df['education_hours'] = df['education_num'] * df['hours_per_week']

    # 5. Бинарный признак брака
    married_categories = ['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse']
    df['is_married'] = df['marital_status'].isin(married_categories).astype(int)

    print(f"После feature engineering: {df.shape}")
    return df

def encode_and_scale(df):
    """Кодирование категориальных + масштабирование"""
    df = df.copy()

    # Категориальные колонки для OneHot
    categorical_cols = ['workclass', 'education', 'marital_status', 'occupation',
                        'relationship', 'race', 'sex', 'native_country', 'age_group']

    # OneHot Encoding
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))

    # Label Encoding для income
    le = LabelEncoder()
    df['income'] = le.fit_transform(df['income'])  # <=50K -> 0, >50K -> 1

    # total_capital -> бинарный (0 если <= 0, иначе 1)
    df['total_capital'] = (df['total_capital'] > 0).astype(int)

    # Числовые колонки для масштабирования
    numeric_cols = ['age', 'fnlwgt', 'education_num', 'capital_gain',
                    'capital_loss', 'hours_per_week', 'hours_per_day',
                    'education_hours', 'is_married', 'total_capital']
    numeric_cols = [col for col in numeric_cols if col in df.columns]

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Объединяем всё
    df = pd.concat([df[numeric_cols], encoded_df, df['income']], axis=1)

    print(f"После кодирования: {df.shape} колонок")
    return df

def train_model(df):
    """Обучение Logistic Regression + MLflow"""
    X = df.drop('income', axis=1)
    y = df['income']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=42, stratify=y)

    # Обучение
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # Предсказания
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Метрики
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"\n=== Метрики модели ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

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

        # Сохранение модели
        mlflow.sklearn.log_model(model, "model")
        joblib.dump(model, "model_income.pkl")
        joblib.dump(X.columns.tolist(), "feature_columns.pkl")

    print("\nМодель сохранена в model_income.pkl")
    print("Метрики залогированы в MLflow")

    return True

# Для Airflow - обёртки
def run_load():
    df = load_data()
    df.to_csv('clean_step1.csv', index=False)
    return True

def run_clean():
    df = pd.read_csv('clean_step1.csv', na_values=['?', ' ?'])
    df = clean_data(df)
    df.to_csv('clean_step2.csv', index=False)
    return True

def run_features():
    df = pd.read_csv('clean_step2.csv')
    df = feature_engineering(df)
    df.to_csv('clean_step3.csv', index=False)
    return True

def run_encode():
    df = pd.read_csv('clean_step3.csv')
    df = encode_and_scale(df)
    df.to_csv('processed_adult.csv', index=False)
    return True

def run_train():
    df = pd.read_csv('processed_adult.csv')
    train_model(df)
    return True