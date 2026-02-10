import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class ChurnPredictorFinal:
    
    def __init__(self, model_path='models/best_rf_model.joblib'):
        # Cargar modelo
        self.model = joblib.load(model_path)
        
        # Columnas EXACTAS que espera tu modelo (COPIADAS DE TU OUTPUT)
        self.expected_columns = [
            'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'MonthlyCharges',
            'TotalCharges', 'num_services', 'tenure_group', 'charge_ratio', 'is_senior',
            'gender_Male', 'MultipleLines_No phone service', 'MultipleLines_Yes',
            'InternetService_Fiber optic', 'InternetService_No', 'Contract_One year',
            'Contract_Two year', 'PaymentMethod_Credit card (automatic)',
            'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
        ]
        
        print(f"✅ Modelo cargado. Espera {len(self.expected_columns)} columnas")
        print(f"📋 Ejemplo: {self.expected_columns[:3]}...")
    
    def _convert_to_numeric(self, value):
        """Convertir seguro a numérico"""
        try:
            return float(value)
        except:
            # Mapear texto a números
            text_map = {
                'yes': 1, 'no': 0, 'male': 1, 'female': 0,
                'true': 1, 'false': 0, '1': 1, '0': 0
            }
            lower_val = str(value).strip().lower()
            return text_map.get(lower_val, 0)
    
    def prepare_features(self, customer_data):

        # Convertir a DataFrame si es dict
        if isinstance(customer_data, dict):
            df = pd.DataFrame([customer_data])
        else:
            df = customer_data.copy()
        
        # Inicializar DataFrame vacío con las columnas correctas
        features = pd.DataFrame(0, index=range(len(df)), columns=self.expected_columns)
        

        # 1. VALORES DIRECTOS (si están en los datos)
   
        # Columnas que pueden venir directamente
        direct_mappings = {
            'tenure': 'tenure',
            'MonthlyCharges': 'MonthlyCharges',
            'TotalCharges': 'TotalCharges',
            'SeniorCitizen': 'SeniorCitizen',
            'Partner': 'Partner',
            'Dependents': 'Dependents'
        }
        
        for source, target in direct_mappings.items():
            if source in df.columns:
                features[target] = df[source].apply(self._convert_to_numeric)
        

        # 2. SERVICIOS (convertir Yes/No a 1/0)
        
        service_cols = {
            'PhoneService': 'PhoneService',
            'OnlineSecurity': 'OnlineSecurity',
            'OnlineBackup': 'OnlineBackup',
            'DeviceProtection': 'DeviceProtection',
            'TechSupport': 'TechSupport',
            'StreamingTV': 'StreamingTV',
            'StreamingMovies': 'StreamingMovies',
            'PaperlessBilling': 'PaperlessBilling'
        }
        
        for source, target in service_cols.items():
            if source in df.columns:
                features[target] = df[source].apply(
                    lambda x: 1 if str(x).strip().lower() in ['yes', '1', 'true'] else 0
                )
        

        # 3. CARACTERÍSTICAS CALCULADAS
        
        # num_services (suma de servicios adicionales)
        additional_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                             'TechSupport', 'StreamingTV', 'StreamingMovies']
        features['num_services'] = features[additional_services].sum(axis=1)
        
        # tenure_group (basado en tenure)
        if 'tenure' in features.columns:
            # Asegurar que tenure es numérico
            features['tenure'] = pd.to_numeric(features['tenure'], errors='coerce').fillna(0)
            
            # Crear grupos (igual que en entrenamiento)
            bins = [0, 12, 24, 36, 48, 60, 72, float('inf')]
            labels = [0, 1, 2, 3, 4, 5, 6]  # Códigos numéricos
            features['tenure_group'] = pd.cut(
                features['tenure'], 
                bins=bins, 
                labels=labels,
                include_lowest=True
            ).astype(int)
        
        # charge_ratio (MonthlyCharges / TotalCharges)
        if 'MonthlyCharges' in features.columns and 'TotalCharges' in features.columns:
            features['charge_ratio'] = features.apply(
                lambda row: row['MonthlyCharges'] / row['TotalCharges'] 
                if row['TotalCharges'] > 0 else 0,
                axis=1
            )
        
        # is_senior (igual que SeniorCitizen)
        features['is_senior'] = features['SeniorCitizen']
        

        # 4. VARIABLES CATEGÓRICAS (ONE-HOT)
        
        # Gender 
        if 'gender' in df.columns:
            features['gender_Male'] = df['gender'].apply(
                lambda x: 1 if str(x).strip().lower() == 'male' else 0
            )
        
        # MultipleLines
        if 'MultipleLines' in df.columns:
            features['MultipleLines_Yes'] = df['MultipleLines'].apply(
                lambda x: 1 if str(x).strip().lower() == 'yes' else 0
            )
            features['MultipleLines_No phone service'] = df['MultipleLines'].apply(
                lambda x: 1 if str(x).strip().lower() == 'no phone service' else 0
            )
        
        # InternetService
        if 'InternetService' in df.columns:
            features['InternetService_Fiber optic'] = df['InternetService'].apply(
                lambda x: 1 if str(x).strip().lower() == 'fiber optic' else 0
            )
            features['InternetService_No'] = df['InternetService'].apply(
                lambda x: 1 if str(x).strip().lower() == 'no' else 0
            )
        else:
            # Si no hay InternetService, asumir DSL (que sería 0 en ambos)
            pass
        
        # Contract
        if 'Contract' in df.columns:
            features['Contract_One year'] = df['Contract'].apply(
                lambda x: 1 if str(x).strip().lower() == 'one year' else 0
            )
            features['Contract_Two year'] = df['Contract'].apply(
                lambda x: 1 if str(x).strip().lower() == 'two year' else 0
            )
        # Month-to-month sería 0 en ambas
        
        # PaymentMethod
        if 'PaymentMethod' in df.columns:
            features['PaymentMethod_Electronic check'] = df['PaymentMethod'].apply(
                lambda x: 1 if str(x).strip().lower() == 'electronic check' else 0
            )
            features['PaymentMethod_Mailed check'] = df['PaymentMethod'].apply(
                lambda x: 1 if str(x).strip().lower() == 'mailed check' else 0
            )
            features['PaymentMethod_Credit card (automatic)'] = df['PaymentMethod'].apply(
                lambda x: 1 if str(x).strip().lower() == 'credit card (automatic)' else 0
            )
        # Bank transfer sería 0 en todas
        

        # 5. VERIFICACIÓN FINAL

        # Asegurar que todas las columnas existen
        for col in self.expected_columns:
            if col not in features.columns:
                print(f"⚠️  Columna faltante: {col} (se pondrá a 0)")
                features[col] = 0
        
        # Ordenar columnas EXACTAMENTE como el modelo las espera
        features = features[self.expected_columns]
        
        # Verificar tipos de datos
        for col in features.columns:
            features[col] = pd.to_numeric(features[col], errors='coerce').fillna(0)
        
        return features
    
    def predict(self, customer_data, verbose=True, threshold=0.35):
        if verbose:
            print("🔧 Preparando features...")
        
        # Preparar features
        features = self.prepare_features(customer_data)
        
        if verbose:
            print(f"✅ Features preparadas: {features.shape[1]} columnas")
            print(f"📋 Primeras columnas: {list(features.columns)[:3]}...")
        
        # Verificar que tenemos todas las columnas
        if list(features.columns) != self.expected_columns:
            if verbose:
                print("❌ ERROR: Las columnas no coinciden")
                print(f"   Esperadas (primeras 3): {self.expected_columns[:3]}")
                print(f"   Obtenidas (primeras 3): {list(features.columns)[:3]}")
            return {
                'success': False,
                'error': 'Column mismatch',
                'expected_columns': self.expected_columns,
                'got_columns': list(features.columns)
            }
        
        # Realizar predicción
        try:
            # Probabilidad
            probabilities = self.model.predict_proba(features)
            churn_probability = float(probabilities[0][1])
            
            # Predicción binaria CON UMBRAL ÓPTIMO 0.35
            churn_prediction = bool(churn_probability >= threshold)
            
            # Determinar nivel de riesgo OPTIMIZADO
            if churn_probability > 0.6:
                risk_level = 'Alto'
                recommendation = '🚨 CONTACTO INMEDIATO: Ofrecer descuento 30% + upgrade gratis'
                color = '🔴'
            elif churn_probability > 0.4:
                risk_level = 'Moderado-Alto'
                recommendation = '📞 Contactar esta semana: Oferta promocional 20%'
                color = '🟠'
            elif churn_probability > 0.3:
                risk_level = 'Moderado'
                recommendation = '📋 Incluir en campaña de retención mensual'
                color = '🟡'
            elif churn_probability > 0.2:
                risk_level = 'Bajo'
                recommendation = '👀 Monitorear trimestralmente'
                color = '🟢'
            else:
                risk_level = 'Muy bajo'
                recommendation = '✅ Cliente estable'
                color = '🔵'
            
            if verbose:
                print(f"🎯 Predicción completada (umbral óptimo: {threshold})")
                print(f"   Probabilidad: {churn_probability:.2%}")
                print(f"   Predicción: {color} {'SÍ' if churn_prediction else 'NO'}")
                print(f"   Nivel de riesgo: {risk_level}")
                print(f"   Recomendación: {recommendation}")
            
            return {
                'success': True,
                'churn_probability': churn_probability,
                'churn_prediction': churn_prediction,
                'risk_level': risk_level,
                'recommendation': recommendation,
                'threshold_used': threshold,
                'color_indicator': color,
                'features_used': len(features.columns),
                'model_performance': {
                    'expected_accuracy': '79%',
                    'expected_precision': '72%',
                    'expected_recall': '94%'
                }
            }
            
        except Exception as e:
            if verbose:
                print(f"❌ Error en predicción: {str(e)}")
            
            return {
                'success': False,
                'error': str(e),
                'churn_probability': 0.5,
                'churn_prediction': False,
                'risk_level': 'Error',
                'recommendation': 'Revisar datos de entrada'
            }

    # EVALUACIÓN DE RENDIMIENTO
 
    def evaluate_model_performance(self, n_samples=20, random_state=42):
        """
        Evaluar el rendimiento del modelo en múltiples clientes
        
        Args:
            n_samples: número de clientes a evaluar
            random_state: semilla para reproducibilidad
            
        Returns:
            DataFrame con resultados y accuracy
        """
        print("\n" + "=" * 60)
        print(f"EVALUACIÓN DEL MODELO ({n_samples} clientes aleatorios)")
        print("=" * 60)
        
        # Cargar datos originales
        df_original = pd.read_csv('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
        
        # Tomar muestra aleatoria
        sample_size = min(n_samples, len(df_original))
        sample = df_original.sample(n=sample_size, random_state=random_state)
        
        results = []
        
        print(f"🔍 Analizando {sample_size} clientes...")
        
        for idx, row in sample.iterrows():
            customer = row.to_dict()
            actual_churn = customer['Churn'] == 'Yes'
            
            # Predecir
            prediction_result = self.predict(customer, verbose=False)
            
            if prediction_result['success']:
                predicted_churn = prediction_result['churn_prediction']
                probability = prediction_result['churn_probability']
                
                correct = (predicted_churn == actual_churn)
                
                results.append({
                    'ID': customer.get('customerID', 'N/A')[:10],  # Solo primeros 10 chars
                    'Actual': 'Yes' if actual_churn else 'No',
                    'Predicted': 'Yes' if predicted_churn else 'No',
                    'Probability': f"{probability:.1%}",
                    'Correct': '✅' if correct else '❌',
                    'Tenure': customer.get('tenure', 'N/A'),
                    'Contract': customer.get('Contract', 'N/A'),
                    'MonthlyCharges': f"${customer.get('MonthlyCharges', 'N/A')}",
                    'InternetService': customer.get('InternetService', 'N/A')
                })
        
        # Crear DataFrame con resultados
        results_df = pd.DataFrame(results)
        
        if len(results_df) > 0:
            print(f"\n📊 RESULTADOS DETALLADOS:")
            print(results_df[['ID', 'Tenure', 'Contract', 'Actual', 'Predicted', 'Probability', 'Correct']].to_string(index=False))
            
            # Calcular métricas
            total = len(results_df)
            correct = (results_df['Correct'] == '✅').sum()
            accuracy = correct / total if total > 0 else 0
            
            # Clientes con churn
            churn_customers = results_df[results_df['Actual'] == 'Yes']
            if len(churn_customers) > 0:
                churn_correct = (churn_customers['Correct'] == '✅').sum()
                churn_accuracy = churn_correct / len(churn_customers)
            else:
                churn_accuracy = 0
            
            # Clientes sin churn
            no_churn_customers = results_df[results_df['Actual'] == 'No']
            if len(no_churn_customers) > 0:
                no_churn_correct = (no_churn_customers['Correct'] == '✅').sum()
                no_churn_accuracy = no_churn_correct / len(no_churn_customers)
            else:
                no_churn_accuracy = 0
            
            print(f"\n📈 MÉTRICAS DEL MODELO:")
            print(f"   Precisión total: {accuracy:.1%} ({correct}/{total})")
            
            if len(churn_customers) > 0:
                print(f"   Precisión en churn detectado: {churn_accuracy:.1%} ({churn_correct}/{len(churn_customers)})")
            
            if len(no_churn_customers) > 0:
                print(f"   Precisión en no churn: {no_churn_accuracy:.1%} ({no_churn_correct}/{len(no_churn_customers)})")
            
            print(f"   Clientes analizados: {total}")
            
            # Matriz de confusión simple
            print(f"\n🎯 MATRIZ DE CONFUSIÓN:")
            tp = len([r for r in results if r['Actual'] == 'Yes' and r['Predicted'] == 'Yes'])
            fp = len([r for r in results if r['Actual'] == 'No' and r['Predicted'] == 'Yes'])
            fn = len([r for r in results if r['Actual'] == 'Yes' and r['Predicted'] == 'No'])
            tn = len([r for r in results if r['Actual'] == 'No' and r['Predicted'] == 'No'])
            
            print(f"                     Predicción")
            print(f"                   Sí         No")
            print(f"   Real   Sí    {tp:3d}        {fn:3d}")
            print(f"          No    {fp:3d}        {tn:3d}")
            
            # Calcular métricas adicionales
            if tp + fp > 0:
                precision = tp / (tp + fp)
                print(f"\n   Precisión (Precision): {precision:.1%}")
            
            if tp + fn > 0:
                recall = tp / (tp + fn)
                print(f"   Recall (Sensibilidad): {recall:.1%}")
            
            if precision > 0 and recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
                print(f"   F1-Score: {f1:.1%}")
            
            # Guardar evaluación
            os.makedirs('reports', exist_ok=True)
            results_df.to_csv(f'reports/model_evaluation_{n_samples}_samples.csv', index=False)
            print(f"\n💾 Resultados guardados en: reports/model_evaluation_{n_samples}_samples.csv")
            
            return results_df, accuracy
        else:
            print("❌ No se pudieron procesar los clientes")
            return pd.DataFrame(), 0

    def evaluate_optimized(self, n_samples=50):
        """
        Evaluación con umbral óptimo 0.35
        """
        print("\n" + "=" * 70)
        print("EVALUACIÓN CON UMBRAL ÓPTIMO (0.35)")
        print("=" * 70)
        
        # Cargar datos
        df_original = pd.read_csv('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
        
        # Muestra balanceada
        churn_samples = df_original[df_original['Churn'] == 'Yes'].sample(
            n=min(n_samples//2, len(df_original[df_original['Churn'] == 'Yes'])), 
            random_state=42
        )
        no_churn_samples = df_original[df_original['Churn'] == 'No'].sample(
            n=min(n_samples//2, len(df_original[df_original['Churn'] == 'No'])), 
            random_state=42
        )
        sample = pd.concat([churn_samples, no_churn_samples])
        
        print(f"📊 Muestra: {len(churn_samples)} con churn + {len(no_churn_samples)} sin churn")
        print(f"🎯 Umbral usado: 0.35 (óptimo según análisis)")
        
        results = []
        
        for idx, row in sample.iterrows():
            customer = row.to_dict()
            actual_churn = customer['Churn'] == 'Yes'
            
            # Predecir con umbral 0.35
            prediction_result = self.predict(customer, verbose=False, threshold=0.35)
            
            if prediction_result['success']:
                predicted_churn = prediction_result['churn_prediction']
                probability = prediction_result['churn_probability']
                
                correct = (predicted_churn == actual_churn)
                
                results.append({
                    'ID': customer.get('customerID', 'N/A')[:8],
                    'Actual': 'Yes' if actual_churn else 'No',
                    'Predicted': 'Yes' if predicted_churn else 'No',
                    'Probability': f"{probability:.1%}",
                    'Correct': '✅' if correct else '❌',
                    'Tenure': customer.get('tenure', 'N/A'),
                    'Contract': customer.get('Contract', 'N/A'),
                    'Risk': prediction_result['risk_level']
                })
        
        # Crear DataFrame
        results_df = pd.DataFrame(results)
        
        if len(results_df) > 0:
            print(f"\n📊 RESULTADOS:")
            print(results_df[['ID', 'Tenure', 'Contract', 'Actual', 'Predicted', 'Probability', 'Correct', 'Risk']].to_string(index=False))
            
            # Métricas
            total = len(results_df)
            correct = (results_df['Correct'] == '✅').sum()
            accuracy = correct / total
            
            # Matriz de confusión
            tp = ((results_df['Actual'] == 'Yes') & (results_df['Predicted'] == 'Yes')).sum()
            fp = ((results_df['Actual'] == 'No') & (results_df['Predicted'] == 'Yes')).sum()
            fn = ((results_df['Actual'] == 'Yes') & (results_df['Predicted'] == 'No')).sum()
            tn = ((results_df['Actual'] == 'No') & (results_df['Predicted'] == 'No')).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"\n📈 MÉTRICAS FINALES (umbral 0.35):")
            print(f"   Accuracy: {accuracy:.1%} ({correct}/{total})")
            print(f"   Precision: {precision:.1%} ({tp}/{tp+fp})")
            print(f"   Recall: {recall:.1%} ({tp}/{tp+fn})")
            print(f"   F1-Score: {f1:.1%}")
            
            print(f"\n🎯 MATRIZ DE CONFUSIÓN:")
            print(f"                     Predicción")
            print(f"                   Sí         No")
            print(f"   Real   Sí    {tp:3d}        {fn:3d}")
            print(f"          No    {fp:3d}        {tn:3d}")
            
            print(f"\n💼 IMPLICACIONES DE NEGOCIO:")
            print(f"   • Se detectarán el {recall:.0%} de los clientes que realmente se van")
            print(f"   • De cada 10 alertas, {precision:.0%} serán clientes que realmente se van")
            print(f"   • Costo de falsos positivos: {fp} clientes contactados innecesariamente")
            print(f"   • Beneficio: Se retendrán aproximadamente {int(tp * 0.7)} clientes (asumiendo 70% de efectividad en retención)")
            
            # Guardar
            results_df.to_csv('reports/final_evaluation_optimal_threshold.csv', index=False)
            
            return results_df, accuracy, precision, recall, f1
        
        return pd.DataFrame(), 0, 0, 0, 0


# FUNCIONES DE PRUEBA Y DEMO

def create_test_customer_high_risk():
    """Cliente de ALTO riesgo (basado en análisis EDA)"""
    return {
        'tenure': 1,                    # Poco tiempo
        'MonthlyCharges': 89.99,        # Cargo alto
        'TotalCharges': 89.99,          # Igual que monthly (solo un mes)
        'SeniorCitizen': 0,
        'Partner': 'No',
        'Dependents': 'No',
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',  # Fiber tiene alto churn
        'OnlineSecurity': 'No',          # Sin servicios de seguridad
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'Yes',           # Pero tiene streaming
        'StreamingMovies': 'Yes',
        'Contract': 'Month-to-month',   # Contrato mensual (alto riesgo)
        'PaperlessBilling': 'Yes',      # Factura electrónica
        'PaymentMethod': 'Electronic check',  # Método con alto churn
        'gender': 'Male'
    }

def create_test_customer_low_risk():
    """Cliente de BAJO riesgo"""
    return {
        'tenure': 60,                   # Mucho tiempo
        'MonthlyCharges': 29.99,        # Cargo bajo
        'TotalCharges': 1799.40,        # 60 * 29.99
        'SeniorCitizen': 1,             # Senior (puede ser estable)
        'Partner': 'Yes',
        'Dependents': 'Yes',
        'PhoneService': 'Yes',
        'MultipleLines': 'Yes',
        'InternetService': 'DSL',       # DSL tiene menos churn
        'OnlineSecurity': 'Yes',        # Con servicios de seguridad
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'Yes',
        'TechSupport': 'Yes',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'Yes',
        'Contract': 'Two year',         # Contrato largo (bajo riesgo)
        'PaperlessBilling': 'No',
        'PaymentMethod': 'Credit card (automatic)',  # Pago automático
        'gender': 'Female'
    }

def test_real_customer_from_dataset():
    """Probar con un cliente real del dataset"""
    print("\n" + "=" * 60)
    print("PRUEBA CON CLIENTE REAL ALEATORIO")
    print("=" * 60)
    
    # Cargar dataset original
    df_original = pd.read_csv('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    # Tomar un cliente ALEATORIO que SÍ hizo churn
    churn_customers = df_original[df_original['Churn'] == 'Yes']
    churn_customer = churn_customers.sample(n=1, random_state=np.random.randint(1000)).iloc[0].to_dict()
    
    # Tomar un cliente ALEATORIO que NO hizo churn
    no_churn_customers = df_original[df_original['Churn'] == 'No']
    no_churn_customer = no_churn_customers.sample(n=1, random_state=np.random.randint(1000)).iloc[0].to_dict()
    
    predictor = ChurnPredictorFinal()
    
    print(f"\n🔍 Cliente CON churn real (aleatorio):")
    print(f"   ID: {churn_customer.get('customerID', 'N/A')}")
    print(f"   Contrato: {churn_customer.get('Contract')}")
    print(f"   Antigüedad: {churn_customer.get('tenure')} meses")
    print(f"   Cargo mensual: ${churn_customer.get('MonthlyCharges', 'N/A')}")
    print(f"   Internet: {churn_customer.get('InternetService', 'N/A')}")
    
    result1 = predictor.predict(churn_customer, verbose=False)
    
    if result1['success']:
        print(f"   📊 Predicción del modelo:")
        print(f"      Probabilidad: {result1['churn_probability']:.2%}")
        print(f"      Predicción: {'SÍ' if result1['churn_prediction'] else 'NO'}")
        print(f"      Riesgo: {result1['risk_level']}")
        print(f"      ¿Acertó?: {'✅ SÍ' if result1['churn_prediction'] else '❌ NO'}")
        print(f"      Confianza: {'Alta' if abs(result1['churn_probability'] - 0.5) > 0.3 else 'Media' if abs(result1['churn_probability'] - 0.5) > 0.15 else 'Baja'}")
    
    print(f"\n" + "-" * 50)
    print(f"🔍 Cliente SIN churn real (aleatorio):")
    print(f"   ID: {no_churn_customer.get('customerID', 'N/A')}")
    print(f"   Contrato: {no_churn_customer.get('Contract')}")
    print(f"   Antigüedad: {no_churn_customer.get('tenure')} meses")
    print(f"   Cargo mensual: ${no_churn_customer.get('MonthlyCharges', 'N/A')}")
    print(f"   Internet: {no_churn_customer.get('InternetService', 'N/A')}")
    
    result2 = predictor.predict(no_churn_customer, verbose=False)
    
    if result2['success']:
        print(f"   📊 Predicción del modelo:")
        print(f"      Probabilidad: {result2['churn_probability']:.2%}")
        print(f"      Predicción: {'SÍ' if result2['churn_prediction'] else 'NO'}")
        print(f"      Riesgo: {result2['risk_level']}")
        print(f"      ¿Acertó?: {'✅ SÍ' if not result2['churn_prediction'] else '❌ NO'}")
        print(f"      Confianza: {'Alta' if abs(result2['churn_probability'] - 0.5) > 0.3 else 'Media' if abs(result2['churn_probability'] - 0.5) > 0.15 else 'Baja'}")
    
    # Estadísticas de rendimiento
    print(f"\n" + "=" * 50)
    print(f"📈 ESTADÍSTICAS DE RENDIMIENTO:")
    
    correct_predictions = 0
    total_predictions = 0
    
    if result1['success']:
        total_predictions += 1
        if result1['churn_prediction']:
            correct_predictions += 1
    
    if result2['success']:
        total_predictions += 1
        if not result2['churn_prediction']:
            correct_predictions += 1
    
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        print(f"   Precisión en esta prueba: {accuracy:.0%} ({correct_predictions}/{total_predictions})")
    
    return result1, result2

def interactive_demo():
    """Demo interactiva para probar diferentes clientes"""
    print("\n" + "=" * 60)
    print("DEMO INTERACTIVA - PREDICCIÓN DE CHURN")
    print("=" * 60)
    
    predictor = ChurnPredictorFinal()
    
    while True:
        print("\n📋 OPCIONES:")
        print("  1. Cliente de ALTO riesgo (nuevo con Fiber + mes a mes)")
        print("  2. Cliente de BAJO riesgo (antiguo con contrato 2 años)")
        print("  3. Ingresar datos manualmente")
        print("  4. Probar con cliente real del dataset")
        print("  5. Evaluar rendimiento del modelo (n clientes)")
        print("  6. Salir")
        
        choice = input("\n👉 Selecciona una opción (1-6): ").strip()
        
        if choice == '1':
            customer = create_test_customer_high_risk()
            print(f"\n🔍 Analizando cliente de ALTO riesgo...")
            print(f"   - Contrato: Month-to-month")
            print(f"   - Internet: Fiber optic")
            print(f"   - Antigüedad: 1 mes")
            
            result = predictor.predict(customer)
            
            if result['success']:
                print(f"\n🎯 RESULTADO:")
                print(f"   Probabilidad de churn: {result['churn_probability']:.2%}")
                print(f"   Predicción: {'✅ SÍ' if result['churn_prediction'] else '✅ NO'}")
                print(f"   Nivel de riesgo: {result['risk_level']}")
                print(f"   Recomendación: {result['recommendation']}")
            else:
                print(f"\n❌ Error: {result.get('error', 'Desconocido')}")
                
        elif choice == '2':
            customer = create_test_customer_low_risk()
            print(f"\n🔍 Analizando cliente de BAJO riesgo...")
            print(f"   - Contrato: Two year")
            print(f"   - Internet: DSL")
            print(f"   - Antigüedad: 60 meses")
            
            result = predictor.predict(customer)
            
            if result['success']:
                print(f"\n🎯 RESULTADO:")
                print(f"   Probabilidad de churn: {result['churn_probability']:.2%}")
                print(f"   Predicción: {'✅ SÍ' if result['churn_prediction'] else '✅ NO'}")
                print(f"   Nivel de riesgo: {result['risk_level']}")
                print(f"   Recomendación: {result['recommendation']}")
            else:
                print(f"\n❌ Error: {result.get('error', 'Desconocido')}")
                
        elif choice == '3':
            print("\n📝 Ingresa los datos del cliente:")
            customer = {}
            customer['tenure'] = int(input("  Antigüedad (meses): ") or "12")
            customer['MonthlyCharges'] = float(input("  Cargo mensual ($): ") or "70.0")
            customer['TotalCharges'] = float(input("  Cargo total ($): ") or str(customer['tenure'] * customer['MonthlyCharges']))
            customer['Contract'] = input("  Contrato (Month-to-month/One year/Two year): ") or "Month-to-month"
            customer['InternetService'] = input("  Internet (DSL/Fiber optic/No): ") or "Fiber optic"
            customer['PaymentMethod'] = input("  Método pago (Electronic check/Mailed check/Bank transfer/Credit card): ") or "Electronic check"
            customer['gender'] = input("  Género (Male/Female): ") or "Male"
            
            result = predictor.predict(customer)
            
            if result['success']:
                print(f"\n🎯 RESULTADO:")
                print(f"   Probabilidad de churn: {result['churn_probability']:.2%}")
                print(f"   Predicción: {'✅ SÍ' if result['churn_prediction'] else '✅ NO'}")
                print(f"   Nivel de riesgo: {result['risk_level']}")
                print(f"   Recomendación: {result['recommendation']}")
            else:
                print(f"\n❌ Error: {result.get('error', 'Desconocido')}")
                
        elif choice == '4':
            test_real_customer_from_dataset()
            
        elif choice == '5':
            # Evaluar rendimiento
            try:
                n_samples = input("  ¿Cuántos clientes evaluar? (default: 20): ").strip()
                n_samples = int(n_samples) if n_samples.isdigit() else 20
                
                results_df, accuracy = predictor.evaluate_model_performance(
                    n_samples=n_samples, 
                    random_state=np.random.randint(1000)
                )
                
                if accuracy > 0:
                    print(f"\n🎯 CONCLUSIÓN:")
                    if accuracy > 0.8:
                        print(f"   ✅ Excelente rendimiento ({accuracy:.1%})")
                    elif accuracy > 0.7:
                        print(f"   👍 Buen rendimiento ({accuracy:.1%})")
                    elif accuracy > 0.6:
                        print(f"   ⚠️  Rendimiento aceptable ({accuracy:.1%})")
                    else:
                        print(f"   🔧 Necesita mejora ({accuracy:.1%})")
            except Exception as e:
                print(f"❌ Error en evaluación: {str(e)}")
                
        elif choice == '6':
            print("\n👋 ¡Hasta luego!")
            break
            
        else:
            print("❌ Opción no válida")

def batch_predict_example():
    """Ejemplo de predicción por lotes"""
    print("\n" + "=" * 60)
    print("PREDICCIÓN POR LOTES - EJEMPLO")
    print("=" * 60)
    
    # Crear 3 clientes de ejemplo
    customers = [
        create_test_customer_high_risk(),
        create_test_customer_low_risk(),
        {
            'tenure': 24,
            'MonthlyCharges': 50.0,
            'TotalCharges': 1200.0,
            'Contract': 'One year',
            'InternetService': 'DSL',
            'PaymentMethod': 'Credit card (automatic)',
            'gender': 'Female'
        }
    ]
    
    predictor = ChurnPredictorFinal()
    
    print(f"\n🔍 Procesando {len(customers)} clientes...")
    
    results = []
    for i, customer in enumerate(customers, 1):
        print(f"\n  Cliente {i}:")
        print(f"    Contrato: {customer.get('Contract', 'N/A')}")
        print(f"    Antigüedad: {customer.get('tenure', 'N/A')} meses")
        
        result = predictor.predict(customer, verbose=False)
        
        if result['success']:
            results.append({
                'Cliente': i,
                'Contrato': customer.get('Contract', 'N/A'),
                'Antigüedad': customer.get('tenure', 'N/A'),
                'Probabilidad': f"{result['churn_probability']:.2%}",
                'Predicción': 'SÍ' if result['churn_prediction'] else 'NO',
                'Riesgo': result['risk_level']
            })
            print(f"    Probabilidad: {result['churn_probability']:.2%}")
            print(f"    Riesgo: {result['risk_level']}")
    
    # Mostrar resumen
    print(f"\n📊 RESUMEN:")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Guardar resultados
    os.makedirs('reports', exist_ok=True)
    results_df.to_csv('reports/batch_predictions.csv', index=False)
    print(f"\n💾 Resultados guardados en: reports/batch_predictions.csv")
    
    return results_df

if __name__ == "__main__":
    print("=" * 60)
    print("PREDICTOR DE CHURN - TELECOMUNICACIONES")
    print("=" * 60)
    print("🎯 Modelo: Random Forest (28 features)")
    print("📊 Dataset: Telco Customer Churn (7,043 clientes)")
    print("=" * 60)
    
    # Inicializar predictor
    try:
        predictor = ChurnPredictorFinal()
        print("✅ Predictor inicializado correctamente")
        
        # Menú principal
        print("\n📋 MODO DE EJECUCIÓN:")
        print("  1. Demo interactiva (recomendado)")
        print("  2. Prueba rápida con ejemplos")
        print("  3. Predicción por lotes")
        print("  4. Evaluar modelo (20 clientes)")
        
        mode = input("\n👉 Selecciona modo (1-4): ").strip()
        
        if mode == '1':
            interactive_demo()
        elif mode == '2':
            # Prueba rápida
            print("\n" + "=" * 60)
            print("PRUEBA RÁPIDA")
            print("=" * 60)
            
            # Cliente alto riesgo
            print("\n🔍 Cliente ALTO riesgo:")
            high_risk = create_test_customer_high_risk()
            result1 = predictor.predict(high_risk)
            
            if result1['success']:
                print(f"   Probabilidad: {result1['churn_probability']:.2%}")
                print(f"   Riesgo: {result1['risk_level']}")
            
            # Cliente bajo riesgo
            print("\n🔍 Cliente BAJO riesgo:")
            low_risk = create_test_customer_low_risk()
            result2 = predictor.predict(low_risk)
            
            if result2['success']:
                print(f"   Probabilidad: {result2['churn_probability']:.2%}")
                print(f"   Riesgo: {result2['risk_level']}")
            
        elif mode == '3':
            batch_predict_example()
        elif mode == '4':
            # Evaluar modelo
            results_df, accuracy = predictor.evaluate_model_performance(n_samples=20)
            print(f"\n🎯 Accuracy final del modelo: {accuracy:.1%}")
        else:
            print("❌ Modo no válido. Ejecutando demo interactiva...")
            interactive_demo()
        
        print("\n" + "=" * 60)
        print("🎉 PREDICTOR FUNCIONANDO CORRECTAMENTE")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {str(e)}")
        print("\n💡 Asegúrate de que:")
        print("   1. El modelo existe en: models/best_rf_model.joblib")
        print("   2. Ejecutas desde la raíz del proyecto")
        print("   3. Si no tienes el modelo, ejecuta: python train_simple_model.py")
        
    except Exception as e:
        print(f"\n❌ Error inesperado: {type(e).__name__}: {str(e)}")
        print(f"\n🔧 Debug info:")
        print(f"   Directorio: {os.getcwd()}")
        print(f"   Modelo existe: {os.path.exists('models/best_rf_model.joblib')}")