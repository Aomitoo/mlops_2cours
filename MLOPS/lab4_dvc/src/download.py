#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import argparse
import yaml
import sys

def load_params(params_path='params.yaml'):
    """Загрузка параметров из YAML"""
    with open(params_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_data(file_path):
    """Загрузка сырых данных"""
    df = pd.read_csv(file_path, na_values=['?', ' ?'])
    print(f"Загружено данных: {df.shape}", file=sys.stderr)
    return df

def clean_data(df, params):
    """Очистка: пропуски, дубликаты, выбросы"""
    df = df.copy()
    df = df.drop_duplicates()
    df = df.dropna(subset=['workclass', 'occupation', 'native_country', 'income'])
    
    numeric_cols = ['age', 'fnlwgt', 'education_num', 'capital_gain',
                    'capital_loss', 'hours_per_week']
    
    iqr_mult = params['prepare'].get('iqr_multiplier', 1.5)
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            median = df[col].median()
            mask = (df[col] < (Q1 - iqr_mult * IQR)) | (df[col] > (Q3 + iqr_mult * IQR))
            df.loc[mask, col] = median
    
    print(f"После очистки: {df.shape}", file=sys.stderr)
    return df

def feature_engineering(df, params):
    """Создание новых признаков"""
    df = df.copy()
    df['total_capital'] = df['capital_gain'] + df['capital_loss']
    
    age_bins = params['features']['age_bins']
    age_labels = params['features']['age_labels']
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)
    
    df['hours_per_day'] = df['hours_per_week'] / 5
    df['education_hours'] = df['education_num'] * df['hours_per_week']
    
    married_cats = params['features']['married_categories']
    df['is_married'] = df['marital_status'].isin(married_cats).astype(int)
    
    print(f"После feature engineering: {df.shape}", file=sys.stderr)
    return df

def encode_and_scale(df):
    """Кодирование + масштабирование"""
    df = df.copy()
    categorical_cols = ['workclass', 'education', 'marital_status', 'occupation',
                        'relationship', 'race', 'sex', 'native_country', 'age_group']
    
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))
    
    le = LabelEncoder()
    df['income'] = le.fit_transform(df['income'])
    df['total_capital'] = (df['total_capital'] > 0).astype(int)
    
    numeric_cols = ['age', 'fnlwgt', 'education_num', 'capital_gain',
                    'capital_loss', 'hours_per_week', 'hours_per_day',
                    'education_hours', 'is_married', 'total_capital']
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    df = pd.concat([df[numeric_cols], encoded_df, df['income']], axis=1).dropna()
    print(f"После кодирования: {df.shape}", file=sys.stderr)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Путь к raw-данным')
    parser.add_argument('--output', type=str, required=True, help='Путь для сохранения')
    parser.add_argument('--params', type=str, default='params.yaml', help='Путь к params.yaml')
    args = parser.parse_args()
    
    params = load_params(args.params)
    df = load_data(args.input)
    df = clean_data(df, params)
    df = feature_engineering(df, params)
    df = encode_and_scale(df)
    df.to_csv(args.output, index=False)
    print(f"✓ Данные сохранены: {args.output}", file=sys.stderr)
