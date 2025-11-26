"""
Değerlendirme Modülü
Bu modül model değerlendirme metriklerini hesaplar ve görselleştirir.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score, classification_report
)
import os
import json
from datetime import datetime

class ModelEvaluator:
    """
    Model değerlendirme sınıfı
    """
    
    def __init__(self, results_dir='results'):
        self.results_dir = results_dir
        self.evaluation_results = {}
        os.makedirs(results_dir, exist_ok=True)
    
    def calculate_metrics(self, y_true, y_pred, model_name):
        """
        Tüm metrikleri hesapla
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # Per-class metrikler
        try:
            precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
            recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
            f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
            
            metrics['precision_per_class'] = precision_per_class.tolist()
            metrics['recall_per_class'] = recall_per_class.tolist()
            metrics['f1_per_class'] = f1_per_class.tolist()
        except:
            pass
        
        self.evaluation_results[model_name] = metrics
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name, class_names=None):
        """
        Confusion matrix görselleştir
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        
        # Normalize edilmiş confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Heatmap
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names if class_names else range(len(cm)),
                   yticklabels=class_names if class_names else range(len(cm)),
                   cbar_kws={'label': 'Normalized Count'})
        
        plt.title(f'Confusion Matrix - {model_name}', fontsize=16, pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        # Kaydet
        filename = f'confusion_matrix_{model_name.replace(" ", "_")}.png'
        filepath = os.path.join(self.results_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Confusion matrix kaydedildi: {filepath}")
        
        return cm
    
    def evaluate_all_models(self, models_dict, X_val, y_val, X_test, y_test, 
                           class_names=None):
        """
        Tüm modelleri değerlendir
        """
        print("\n" + "=" * 60)
        print("MODEL DEĞERLENDİRME")
        print("=" * 60)
        
        all_metrics = []
        
        for model_name, model in models_dict.items():
            print(f"\n{model_name} değerlendiriliyor...")
            
            # Tahmin yap
            if 'LSTM' in model_name:
                # Keras modelleri için
                try:
                    y_pred_proba = model.predict(X_val, verbose=0)
                    y_pred = np.argmax(y_pred_proba, axis=1)
                    # y_val'nin shape'ini kontrol et
                    if len(y_val.shape) > 1 and y_val.shape[1] > 1:
                        y_true = np.argmax(y_val, axis=1)
                    else:
                        y_true = y_val.flatten() if len(y_val.shape) > 1 else y_val
                except Exception as e:
                    print(f"  ⚠ LSTM tahmin hatası: {e}")
                    continue
            else:
                # Scikit-learn modelleri için
                y_pred = model.predict(X_val)
                y_true = y_val
            
            # Metrikleri hesapla
            metrics = self.calculate_metrics(y_true, y_pred, model_name)
            
            # Confusion matrix
            self.plot_confusion_matrix(y_true, y_pred, model_name, class_names)
            
            # Classification report
            report = classification_report(y_true, y_pred, 
                                         target_names=class_names,
                                         output_dict=True, zero_division=0)
            
            # Sonuçları kaydet
            all_metrics.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score']
            })
            
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1_score']:.4f}")
        
        # Metrikleri DataFrame olarak kaydet
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df = metrics_df.sort_values('F1-Score', ascending=False)
        
        csv_path = os.path.join(self.results_dir, 'evaluation_metrics.csv')
        metrics_df.to_csv(csv_path, index=False)
        print(f"\n✓ Tüm metrikler kaydedildi: {csv_path}")
        
        # Görselleştir
        self.plot_metrics_comparison(metrics_df)
        
        return metrics_df
    
    def plot_metrics_comparison(self, metrics_df):
        """
        Metrik karşılaştırmasını görselleştir
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx // 2, idx % 2]
            ax.barh(metrics_df['Model'], metrics_df[metric], edgecolor='black')
            ax.set_xlabel(metric)
            ax.set_title(f'{metric} Karşılaştırması')
            ax.grid(True, alpha=0.3, axis='x')
            ax.set_xlim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'metrics_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Metrik karşılaştırma grafiği kaydedildi")
    
    def analyze_overfitting(self, models_dict, training_histories):
        """
        Overfitting analizi yap
        """
        print("\n" + "=" * 60)
        print("OVERFITTING ANALİZİ")
        print("=" * 60)
        
        overfitting_analysis = {}
        
        for model_name, history in training_histories.items():
            if history is None:
                continue
            
            # Son epoch değerleri
            train_acc = history['accuracy'][-1] if 'accuracy' in history else None
            val_acc = history['val_accuracy'][-1] if 'val_accuracy' in history else None
            train_loss = history['loss'][-1] if 'loss' in history else None
            val_loss = history['val_loss'][-1] if 'val_loss' in history else None
            
            # Overfitting göstergeleri
            acc_gap = train_acc - val_acc if train_acc and val_acc else None
            loss_gap = val_loss - train_loss if train_loss and val_loss else None
            
            overfitting_analysis[model_name] = {
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
                'accuracy_gap': acc_gap,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'loss_gap': loss_gap,
                'overfitting_risk': 'High' if acc_gap and acc_gap > 0.1 else 'Low' if acc_gap and acc_gap < 0.05 else 'Medium'
            }
            
            print(f"\n{model_name}:")
            if train_acc and val_acc:
                print(f"  Train Accuracy: {train_acc:.4f}")
                print(f"  Val Accuracy: {val_acc:.4f}")
                print(f"  Accuracy Gap: {acc_gap:.4f}")
                print(f"  Overfitting Risk: {overfitting_analysis[model_name]['overfitting_risk']}")
        
        # Overfitting analiz grafikleri
        self.plot_overfitting_analysis(training_histories)
        
        # JSON olarak kaydet
        json_path = os.path.join(self.results_dir, 'overfitting_analysis.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(overfitting_analysis, f, indent=2, default=str)
        print(f"\n✓ Overfitting analizi kaydedildi: {json_path}")
        
        return overfitting_analysis
    
    def plot_overfitting_analysis(self, training_histories):
        """
        Overfitting analiz grafikleri
        """
        for model_name, history in training_histories.items():
            if history is None:
                continue
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            epochs = range(1, len(history['accuracy']) + 1)
            
            # Accuracy grafiği
            axes[0].plot(epochs, history['accuracy'], 'b-', label='Train Accuracy', linewidth=2)
            axes[0].plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
            axes[0].set_title(f'{model_name} - Accuracy (Overfitting Analysis)')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Accuracy')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Loss grafiği
            axes[1].plot(epochs, history['loss'], 'b-', label='Train Loss', linewidth=2)
            axes[1].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
            axes[1].set_title(f'{model_name} - Loss (Overfitting Analysis)')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            filename = f'overfitting_analysis_{model_name.replace(" ", "_")}.png'
            plt.savefig(os.path.join(self.results_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()
        
        print("✓ Overfitting analiz grafikleri kaydedildi")
    
    def select_best_model(self, metrics_df, models_dict, results_dir='results'):
        """
        En iyi modeli seç ve kaydet
        """
        print("\n" + "=" * 60)
        print("EN İYİ MODEL SEÇİMİ")
        print("=" * 60)
        
        # F1-Score'a göre en iyi model
        best_model_name = metrics_df.iloc[0]['Model']
        best_model = models_dict[best_model_name]
        
        print(f"\n✓ En iyi model: {best_model_name}")
        print(f"  Accuracy: {metrics_df.iloc[0]['Accuracy']:.4f}")
        print(f"  F1-Score: {metrics_df.iloc[0]['F1-Score']:.4f}")
        
        # Models klasörünü oluştur
        models_dir = os.path.join(results_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # Modeli kaydet
        if 'LSTM' in best_model_name:
            model_path = os.path.join(models_dir, 'best_model.h5')
            best_model.save(model_path)
            print(f"✓ En iyi model kaydedildi: {model_path}")
        else:
            import pickle
            model_path = os.path.join(models_dir, 'best_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(best_model, f)
            print(f"✓ En iyi model kaydedildi: {model_path}")
        
        # Model bilgilerini kaydet
        model_info = {
            'model_name': best_model_name,
            'accuracy': float(metrics_df.iloc[0]['Accuracy']),
            'precision': float(metrics_df.iloc[0]['Precision']),
            'recall': float(metrics_df.iloc[0]['Recall']),
            'f1_score': float(metrics_df.iloc[0]['F1-Score']),
            'saved_path': model_path,
            'timestamp': datetime.now().isoformat()
        }
        
        info_path = os.path.join(models_dir, 'best_model_info.json')
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"✓ Model bilgileri kaydedildi: {info_path}")
        
        return best_model_name, best_model, model_info

