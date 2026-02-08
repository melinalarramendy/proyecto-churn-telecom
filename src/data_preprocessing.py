import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
import os

class TelcoDataPreprocessor:
    """Preprocesador para dataset de Telco Customer Churn"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.encoders = {}
        
    def load_data(self, data_path):
        """Cargar datos desde archivo CSV"""
        df = pd.read_csv(data_path)
        print(f"✅ Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
        return df
    
    def initial_clean(self, df):
        """Limpieza inicial del dataset"""
        # Crear copia
        df_clean = df.copy()
        
        # 1. Eliminar customerID (no útil para modelo)
        if 'customerID' in df_clean.columns:
            df_clean = df_clean.drop('customerID', axis=1)
        
        # 2. Convertir TotalCharges a numérico
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
        
        # 3. Imputar valores nulos en TotalCharges
        mask = df_clean['TotalCharges'].isnull()
        df_clean.loc[mask, 'TotalCharges'] = df_clean.loc[mask, 'MonthlyCharges'] * df_clean.loc[mask, 'tenure']
        
        print(f"✅ Limpieza inicial completada. Valores nulos: {df_clean.isnull().sum().sum()}")
        return df_clean
    
    def feature_engineering(self, df):
        """Crear nuevas características"""
        df_fe = df.copy()
        
        # 1. Número total de servicios
        service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                       'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        # Contar servicios adicionales (excluyendo 'No internet service')
        df_fe['num_services'] = 0
        for col in service_cols:
            df_fe['num_services'] += (df_fe[col] == 'Yes').astype(int)
        
        # 2. Grupos de antigüedad
        df_fe['tenure_group'] = pd.cut(df_fe['tenure'], 
                                      bins=[0, 12, 24, 36, 48, 60, 72],
                                      labels=['0-1', '1-2', '2-3', '3-4', '4-5', '5-6'])
        
        # 3. Ratio de cargos
        df_fe['charge_ratio'] = df_fe['MonthlyCharges'] / (df_fe['TotalCharges'] + 1)  # +1 para evitar división por 0
        
        # 4. Es cliente mayor
        df_fe['is_senior'] = df_fe['SeniorCitizen'].map({0: 0, 1: 1})
        
        print(f"✅ Ingeniería de características completada. Nuevas features: {list(df_fe.columns[-4:])}")
        return df_fe
    
    def encode_categorical(self, df):
        """Codificar variables categóricas"""
        df_encoded = df.copy()
        
        # Mapeos específicos
        binary_map = {'Yes': 1, 'No': 0, 'No internet service': 0}
        
        # Columnas para diferentes tipos de encoding
        binary_cols = ['Partner', 'Dependents', 'PhoneService', 
                      'PaperlessBilling', 'Churn']
        
        onehot_cols = ['gender', 'MultipleLines', 'InternetService',
                      'Contract', 'PaymentMethod']
        
        service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                       'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        # 1. Binary encoding
        for col in binary_cols:
            if col in df_encoded.columns:
                df_encoded[col] = df_encoded[col].map(binary_map)
        
        # 2. Service columns (Yes/No/No internet service)
        for col in service_cols:
            if col in df_encoded.columns:
                df_encoded[col] = df_encoded[col].map(binary_map)
        
        # 3. One-Hot Encoding
        df_encoded = pd.get_dummies(df_encoded, columns=onehot_cols, drop_first=True)
        
        # 4. Label encoding para tenure_group
        if 'tenure_group' in df_encoded.columns:
            le = LabelEncoder()
            df_encoded['tenure_group'] = le.fit_transform(df_encoded['tenure_group'])
            self.encoders['tenure_group'] = le
        
        print(f"✅ Codificación completada. Columnas finales: {df_encoded.shape[1]}")
        return df_encoded
    
    def scale_features(self, df):
        """Escalar características numéricas"""
        df_scaled = df.copy()
        
        # Identificar columnas numéricas
        numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
        
        # Excluir columnas binarias y target
        exclude_cols = ['Churn', 'SeniorCitizen', 'Partner', 'Dependents', 
                       'PhoneService', 'PaperlessBilling'] + \
                      [col for col in df_scaled.columns if 'Online' in col or 
                       'Streaming' in col or 'Device' in col or 'Tech' in col]
        
        cols_to_scale = [col for col in numeric_cols if col not in exclude_cols]
        
        # Escalar
        df_scaled[cols_to_scale] = self.scaler.fit_transform(df_scaled[cols_to_scale])
        
        print(f"✅ Escalado completado para {len(cols_to_scale)} columnas")
        return df_scaled
    
    def split_features_target(self, df):
        """Separar features y target"""
        X = df.drop('Churn', axis=1)
        y = df['Churn']
        
        print(f"✅ Separación completada: X={X.shape}, y={y.shape}")
        return X, y
    
    def save_processed_data(self, X, y, path='./data/processed'):
        """Guardar datos procesados"""
        os.makedirs(path, exist_ok=True)
        
        X.to_csv(f'{path}/X_processed.csv', index=False)
        y.to_csv(f'{path}/y_processed.csv', index=False)
        
        # Guardar scaler y encoders
        joblib.dump(self.scaler, f'{path}/scaler.joblib')
        joblib.dump(self.encoders, f'{path}/encoders.joblib')
        
        print(f"✅ Datos guardados en: {path}")
    
    def run_pipeline(self, data_path, save=True):
        """Ejecutar pipeline completo de preprocesamiento"""
        print("=" * 60)
        print("INICIANDO PIPELINE DE PREPROCESAMIENTO")
        print("=" * 60)
        
        # 1. Cargar datos
        df = self.load_data(data_path)
        
        # 2. Limpieza inicial
        df = self.initial_clean(df)
        
        # 3. Ingeniería de características
        df = self.feature_engineering(df)
        
        # 4. Codificación
        df = self.encode_categorical(df)
        
        # 5. Escalado
        df = self.scale_features(df)
        
        # 6. Separar features y target
        X, y = self.split_features_target(df)
        
        # 7. Guardar
        if save:
            self.save_processed_data(X, y)
        
        print("\n✅ PIPELINE COMPLETADO EXITOSAMENTE")
        return X, y

# Para ejecutar desde terminal
if __name__ == "__main__":
    preprocessor = TelcoDataPreprocessor()
    X, y = preprocessor.run_pipeline('./data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')