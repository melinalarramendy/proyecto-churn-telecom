from predict import ChurnPredictorFinal
import pandas as pd
import numpy as np

def analyze_threshold_performance():
    """Análisis completo de rendimiento por umbral"""
    
    print("=" * 70)
    print("ANÁLISIS DE UMBRAL ÓPTIMO - CHURN PREDICTION")
    print("=" * 70)
    
    # Cargar predictor
    predictor = ChurnPredictorFinal()
    
    # Cargar datos
    df_original = pd.read_csv('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    # Tomar muestra balanceada (50% churn, 50% no churn)
    churn_samples = df_original[df_original['Churn'] == 'Yes'].sample(n=50, random_state=42)
    no_churn_samples = df_original[df_original['Churn'] == 'No'].sample(n=50, random_state=42)
    sample = pd.concat([churn_samples, no_churn_samples])
    
    print(f"📊 Muestra: {len(churn_samples)} con churn + {len(no_churn_samples)} sin churn")
    
    # Probar diferentes umbrales
    thresholds = np.arange(0.2, 0.65, 0.05)  # De 20% a 60%
    
    results = []
    
    for threshold in thresholds:
        print(f"\n🔍 Umbral: {threshold:.2f}")
        
        tp = fp = fn = tn = 0
        all_probabilities = []
        
        for idx, row in sample.iterrows():
            customer = row.to_dict()
            actual_churn = customer['Churn'] == 'Yes'
            
            # Predecir
            try:
                # Usar el método predict existente pero forzar threshold
                features = predictor.prepare_features(customer)
                
                if hasattr(predictor.model, 'predict_proba'):
                    probabilities = predictor.model.predict_proba(features)
                    churn_probability = float(probabilities[0][1])
                else:
                    churn_probability = 0.5
                
                predicted_churn = churn_probability >= threshold
                all_probabilities.append(churn_probability)
                
                # Contar
                if actual_churn and predicted_churn:
                    tp += 1
                elif actual_churn and not predicted_churn:
                    fn += 1
                elif not actual_churn and predicted_churn:
                    fp += 1
                else:
                    tn += 1
                    
            except Exception as e:
                continue
        
        # Calcular métricas
        total = tp + fp + fn + tn
        if total > 0:
            accuracy = (tp + tn) / total
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Business metrics
            false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
            detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            results.append({
                'threshold': threshold,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn,
                'false_positive_rate': false_positive_rate,
                'detection_rate': detection_rate,
                'avg_probability': np.mean(all_probabilities) if all_probabilities else 0
            })
            
            print(f"   Accuracy: {accuracy:.1%} | F1: {f1:.1%}")
            print(f"   TP: {tp:2d} | FP: {fp:2d} | FN: {fn:2d} | TN: {tn:2d}")
    
    # Crear DataFrame
    results_df = pd.DataFrame(results)
    
    # Encontrar mejor umbral por F1-Score
    best_f1_idx = results_df['f1'].idxmax()
    best_threshold_f1 = results_df.loc[best_f1_idx, 'threshold']
    
    # Encontrar mejor equilibrio (accuracy decente + recall decente)
    results_df['balanced_score'] = results_df['accuracy'] * 0.4 + results_df['recall'] * 0.6
    best_balanced_idx = results_df['balanced_score'].idxmax()
    best_threshold_balanced = results_df.loc[best_balanced_idx, 'threshold']
    
    # Mostrar resultados
    print(f"\n" + "=" * 70)
    print("🎯 RESULTADOS DEL ANÁLISIS")
    print("=" * 70)
    
    print(f"\n📊 MEJOR UMBRAL POR F1-SCORE: {best_threshold_f1:.2f}")
    best_row_f1 = results_df.loc[best_f1_idx]
    print(f"   Accuracy: {best_row_f1['accuracy']:.1%}")
    print(f"   Precision: {best_row_f1['precision']:.1%}")
    print(f"   Recall: {best_row_f1['recall']:.1%}")
    print(f"   F1-Score: {best_row_f1['f1']:.1%}")
    print(f"   Detección de churn: {best_row_f1['tp']}/{best_row_f1['tp']+best_row_f1['fn']}")
    print(f"   Falsos positivos: {best_row_f1['fp']}/{best_row_f1['fp']+best_row_f1['tn']}")
    
    print(f"\n📊 MEJOR UMBRAL BALANCEADO: {best_threshold_balanced:.2f}")
    best_row_balanced = results_df.loc[best_balanced_idx]
    print(f"   Accuracy: {best_row_balanced['accuracy']:.1%}")
    print(f"   Precision: {best_row_balanced['precision']:.1%}")
    print(f"   Recall: {best_row_balanced['recall']:.1%}")
    print(f"   F1-Score: {best_row_balanced['f1']:.1%}")
    print(f"   Detección de churn: {best_row_balanced['tp']}/{best_row_balanced['tp']+best_row_balanced['fn']}")
    print(f"   Falsos positivos: {best_row_balanced['fp']}/{best_row_balanced['fp']+best_row_balanced['tn']}")
    
    # Recomendación basada en negocio
    print(f"\n💼 RECOMENDACIÓN PARA NEGOCIO:")
    print("-" * 40)
    
    # Si queremos minimizar falsos positivos (no molestar a clientes buenos)
    low_fp_idx = results_df['false_positive_rate'].idxmin()
    low_fp_threshold = results_df.loc[low_fp_idx, 'threshold']
    
    # Si queremos maximizar detección (encontrar todos los churn)
    high_recall_idx = results_df['recall'].idxmax()
    high_recall_threshold = results_df.loc[high_recall_idx, 'threshold']
    
    print(f"   Para MINIMIZAR falsos positivos: {low_fp_threshold:.2f}")
    print(f"     (No molestar a clientes que no se van)")
    
    print(f"   Para MAXIMIZAR detección de churn: {high_recall_threshold:.2f}")
    print(f"     (Encontrar a todos los que se van)")
    
    print(f"   Para EQUILIBRIO óptimo: {best_threshold_balanced:.2f}")
    print(f"     (Balance entre precisión y cobertura)")
    
    # Guardar resultados
    import os
    os.makedirs('reports', exist_ok=True)
    results_df.to_csv('reports/threshold_analysis_detailed.csv', index=False)
    
    print(f"\n💾 Resultados guardados en: reports/threshold_analysis_detailed.csv")
    
    # Crear gráfico
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Gráfico 1: Accuracy y F1
        ax1 = axes[0, 0]
        ax1.plot(results_df['threshold'], results_df['accuracy'], 'b-', label='Accuracy', linewidth=2)
        ax1.plot(results_df['threshold'], results_df['f1'], 'r-', label='F1-Score', linewidth=2)
        ax1.axvline(x=best_threshold_balanced, color='green', linestyle='--', label=f'Óptimo: {best_threshold_balanced:.2f}')
        ax1.set_xlabel('Umbral')
        ax1.set_ylabel('Score')
        ax1.set_title('Accuracy vs F1-Score por Umbral')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: Precision y Recall
        ax2 = axes[0, 1]
        ax2.plot(results_df['threshold'], results_df['precision'], 'g-', label='Precision', linewidth=2)
        ax2.plot(results_df['threshold'], results_df['recall'], 'm-', label='Recall', linewidth=2)
        ax2.axvline(x=best_threshold_balanced, color='green', linestyle='--')
        ax2.set_xlabel('Umbral')
        ax2.set_ylabel('Score')
        ax2.set_title('Precision vs Recall por Umbral')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Gráfico 3: TP, FP, FN, TN
        ax3 = axes[1, 0]
        width = 0.2
        x = np.arange(len(results_df))
        ax3.bar(x - 1.5*width, results_df['tp'], width, label='TP', color='green')
        ax3.bar(x - 0.5*width, results_df['fp'], width, label='FP', color='red')
        ax3.bar(x + 0.5*width, results_df['fn'], width, label='FN', color='orange')
        ax3.bar(x + 1.5*width, results_df['tn'], width, label='TN', color='blue')
        ax3.set_xlabel('Umbral (índice)')
        ax3.set_ylabel('Cantidad')
        ax3.set_title('Matriz de Confusión por Umbral')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Gráfico 4: FPR y Detection Rate
        ax4 = axes[1, 1]
        ax4.plot(results_df['threshold'], results_df['false_positive_rate'], 'r-', label='FPR', linewidth=2)
        ax4.plot(results_df['threshold'], results_df['detection_rate'], 'b-', label='Detection Rate', linewidth=2)
        ax4.axvline(x=best_threshold_balanced, color='green', linestyle='--')
        ax4.set_xlabel('Umbral')
        ax4.set_ylabel('Tasa')
        ax4.set_title('Tasa de Falsos Positivos vs Tasa de Detección')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'Análisis de Umbral Óptimo - Mejor: {best_threshold_balanced:.2f}', fontsize=16)
        plt.tight_layout()
        plt.savefig('reports/threshold_analysis_plots.png', dpi=100, bbox_inches='tight')
        print(f"💾 Gráficos guardados en: reports/threshold_analysis_plots.png")
        
    except ImportError:
        print("⚠️  Matplotlib no disponible para gráficos")
    
    return best_threshold_balanced, results_df

if __name__ == "__main__":
    best_threshold, results = analyze_threshold_performance()
    
    print(f"\n" + "=" * 70)
    print("🎯 ACCIÓN RECOMENDADA:")
    print("=" * 70)
    print(f"1. Usa umbral {best_threshold:.2f} en tu predictor")
    print(f"2. Actualiza el método predict() con threshold={best_threshold:.2f}")
    print(f"3. Vuelve a evaluar con este umbral")
    print(f"\n💡 Para usar: predictor.predict(customer, threshold={best_threshold:.2f})")