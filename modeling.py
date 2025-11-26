"""
Modelleme Modülü
Bu modül farklı ML ve Deep Learning modellerini eğitir ve karşılaştırır.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
import os
import pickle
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# TensorFlow GPU ayarları
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class ModelTrainer:
    """
    Model eğitimi ve karşılaştırma sınıfı
    """
    
    def __init__(self, results_dir='results'):
        self.results_dir = results_dir
        self.models = {}
        self.results = {}
        os.makedirs(results_dir, exist_ok=True)
    
    def train_traditional_models(self, X_train, X_val, y_train, y_val):
        """
        Geleneksel ML modellerini eğit
        """
        print("\n" + "=" * 60)
        print("GELENEKSEL MAKİNE ÖĞRENMESİ MODELLERİ")
        print("=" * 60)
        
        # 1. Logistic Regression
        print("\n1. Logistic Regression Modeli Eğitiliyor...")
        lr_model = LogisticRegression(max_iter=1000, random_state=42, multi_class='ovr')
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_val)
        lr_acc = accuracy_score(y_val, lr_pred)
        lr_f1 = f1_score(y_val, lr_pred, average='weighted')
        
        self.models['Logistic Regression'] = lr_model
        self.results['Logistic Regression'] = {
            'accuracy': lr_acc,
            'f1_score': lr_f1,
            'predictions': lr_pred
        }
        print(f"✓ Logistic Regression - Accuracy: {lr_acc:.4f}, F1-Score: {lr_f1:.4f}")
        
        # 2. Random Forest
        print("\n2. Random Forest Modeli Eğitiliyor...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_val)
        rf_acc = accuracy_score(y_val, rf_pred)
        rf_f1 = f1_score(y_val, rf_pred, average='weighted')
        
        self.models['Random Forest'] = rf_model
        self.results['Random Forest'] = {
            'accuracy': rf_acc,
            'f1_score': rf_f1,
            'predictions': rf_pred
        }
        print(f"✓ Random Forest - Accuracy: {rf_acc:.4f}, F1-Score: {rf_f1:.4f}")
        
        # 3. SVM
        print("\n3. SVM Modeli Eğitiliyor...")
        # SVM için daha küçük bir örneklem kullan (büyük veri setleri için yavaş)
        sample_size = min(5000, len(X_train))
        indices = np.random.choice(len(X_train), sample_size, replace=False)
        X_train_sample = X_train.iloc[indices] if isinstance(X_train, pd.DataFrame) else X_train[indices]
        y_train_sample = y_train.iloc[indices] if isinstance(y_train, pd.Series) else y_train[indices]
        
        svm_model = SVC(kernel='rbf', random_state=42, probability=True)
        svm_model.fit(X_train_sample, y_train_sample)
        svm_pred = svm_model.predict(X_val)
        svm_acc = accuracy_score(y_val, svm_pred)
        svm_f1 = f1_score(y_val, svm_pred, average='weighted')
        
        self.models['SVM'] = svm_model
        self.results['SVM'] = {
            'accuracy': svm_acc,
            'f1_score': svm_f1,
            'predictions': svm_pred
        }
        print(f"✓ SVM - Accuracy: {svm_acc:.4f}, F1-Score: {svm_f1:.4f}")
        
        return self.models, self.results
    
    def prepare_lstm_data(self, X_train, X_val, X_test, y_train, y_val, y_test, sequence_length=1):
        """
        LSTM için veriyi sequence formatına dönüştür
        Tabular veri için her satırı tek bir timestep olarak ele alır
        """
        # Veriyi numpy array'e çevir
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
            X_val = X_val.values
            X_test = X_test.values
        
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
            y_val = y_val.values
            y_test = y_test.values
        
        # Tabular veri için: Her satırı bir sequence olarak ele al
        # Shape: (samples, timesteps, features)
        # sequence_length=1 için: (samples, 1, features)
        n_features = X_train.shape[1] if len(X_train.shape) > 1 else 1
        
        # Reshape: (samples, 1, features) - her örnek tek bir timestep
        X_train_seq = X_train.reshape(-1, sequence_length, n_features)
        X_val_seq = X_val.reshape(-1, sequence_length, n_features)
        X_test_seq = X_test.reshape(-1, sequence_length, n_features)
        
        # One-hot encoding için sınıf sayısını belirle
        n_classes = len(np.unique(np.concatenate([y_train, y_val, y_test])))
        
        # One-hot encoding
        y_train_categorical = keras.utils.to_categorical(y_train, n_classes)
        y_val_categorical = keras.utils.to_categorical(y_val, n_classes)
        y_test_categorical = keras.utils.to_categorical(y_test, n_classes)
        
        print(f"\nLSTM Veri Hazırlama:")
        print(f"  Train shape: {X_train_seq.shape}")
        print(f"  Validation shape: {X_val_seq.shape}")
        print(f"  Test shape: {X_test_seq.shape}")
        print(f"  Features: {n_features}")
        print(f"  Classes: {n_classes}")
        
        return (X_train_seq, y_train_categorical, 
                X_val_seq, y_val_categorical,
                X_test_seq, y_test_categorical, n_classes)
    
    def train_lstm_model(self, X_train, X_val, y_train, y_val, n_features, n_classes, 
                        epochs=50, batch_size=32):
        """
        LSTM modeli eğit
        """
        print("\n" + "=" * 60)
        print("LSTM MODELİ")
        print("=" * 60)
        
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], n_features)),
            Dropout(0.3),
            LSTM(64, return_sequences=False),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(n_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("\nModel Özeti:")
        model.summary()
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.0001
        )
        
        # Model eğitimi
        print("\nModel Eğitiliyor...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Değerlendirme
        val_pred = model.predict(X_val)
        val_pred_classes = np.argmax(val_pred, axis=1)
        val_true_classes = np.argmax(y_val, axis=1)
        
        lstm_acc = accuracy_score(val_true_classes, val_pred_classes)
        lstm_f1 = f1_score(val_true_classes, val_pred_classes, average='weighted')
        
        self.models['LSTM'] = model
        self.results['LSTM'] = {
            'accuracy': lstm_acc,
            'f1_score': lstm_f1,
            'predictions': val_pred_classes,
            'history': history.history
        }
        
        print(f"\n✓ LSTM - Accuracy: {lstm_acc:.4f}, F1-Score: {lstm_f1:.4f}")
        
        # Eğitim geçmişini görselleştir
        self.plot_training_history(history.history, 'LSTM')
        
        return model, history
    
    def train_bidirectional_lstm_model(self, X_train, X_val, y_train, y_val, n_features, n_classes,
                                      epochs=50, batch_size=32):
        """
        Bidirectional LSTM modeli eğit
        """
        print("\n" + "=" * 60)
        print("BIDIRECTIONAL LSTM MODELİ")
        print("=" * 60)
        
        model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True), input_shape=(X_train.shape[1], n_features)),
            Dropout(0.3),
            Bidirectional(LSTM(64, return_sequences=False)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(n_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("\nModel Özeti:")
        model.summary()
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.0001
        )
        
        # Model eğitimi
        print("\nModel Eğitiliyor...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Değerlendirme
        val_pred = model.predict(X_val)
        val_pred_classes = np.argmax(val_pred, axis=1)
        val_true_classes = np.argmax(y_val, axis=1)
        
        bilstm_acc = accuracy_score(val_true_classes, val_pred_classes)
        bilstm_f1 = f1_score(val_true_classes, val_pred_classes, average='weighted')
        
        self.models['Bidirectional LSTM'] = model
        self.results['Bidirectional LSTM'] = {
            'accuracy': bilstm_acc,
            'f1_score': bilstm_f1,
            'predictions': val_pred_classes,
            'history': history.history
        }
        
        print(f"\n✓ Bidirectional LSTM - Accuracy: {bilstm_acc:.4f}, F1-Score: {bilstm_f1:.4f}")
        
        # Eğitim geçmişini görselleştir
        self.plot_training_history(history.history, 'Bidirectional_LSTM')
        
        return model, history
    
    def hyperparameter_optimization(self, X_train, X_val, y_train, y_val):
        """
        Hiperparametre optimizasyonu (Random Forest için)
        """
        print("\n" + "=" * 60)
        print("HİPERPARAMETRE OPTİMİZASYONU")
        print("=" * 60)
        
        # Random Forest için hiperparametre grid'i
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        print("Random Forest için RandomSearchCV uygulanıyor...")
        random_search = RandomizedSearchCV(
            estimator=rf_base,
            param_distributions=param_grid,
            n_iter=20,
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        random_search.fit(X_train, y_train)
        
        print(f"\n✓ En iyi parametreler: {random_search.best_params_}")
        print(f"✓ En iyi skor: {random_search.best_score_:.4f}")
        
        # En iyi modeli kaydet
        best_rf = random_search.best_estimator_
        best_pred = best_rf.predict(X_val)
        best_acc = accuracy_score(y_val, best_pred)
        best_f1 = f1_score(y_val, best_pred, average='weighted')
        
        self.models['Random Forest (Optimized)'] = best_rf
        self.results['Random Forest (Optimized)'] = {
            'accuracy': best_acc,
            'f1_score': best_f1,
            'predictions': best_pred,
            'best_params': random_search.best_params_
        }
        
        print(f"✓ Optimize edilmiş Random Forest - Accuracy: {best_acc:.4f}, F1-Score: {best_f1:.4f}")
        
        return best_rf, random_search.best_params_
    
    def plot_training_history(self, history, model_name):
        """
        Eğitim geçmişini görselleştir
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy
        ax1.plot(history['accuracy'], label='Train Accuracy')
        ax1.plot(history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title(f'{model_name} - Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss
        ax2.plot(history['loss'], label='Train Loss')
        ax2.plot(history['val_loss'], label='Validation Loss')
        ax2.set_title(f'{model_name} - Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'{model_name}_training_history.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def compare_models(self):
        """
        Tüm modelleri karşılaştır
        """
        print("\n" + "=" * 60)
        print("MODEL KARŞILAŞTIRMASI")
        print("=" * 60)
        
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'F1-Score': results['f1_score']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        print("\nModel Performans Karşılaştırması:")
        print(comparison_df.to_string(index=False))
        
        # Görselleştirme
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Accuracy karşılaştırması
        ax1.barh(comparison_df['Model'], comparison_df['Accuracy'], edgecolor='black')
        ax1.set_xlabel('Accuracy')
        ax1.set_title('Model Accuracy Karşılaştırması')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # F1-Score karşılaştırması
        ax2.barh(comparison_df['Model'], comparison_df['F1-Score'], edgecolor='black', color='orange')
        ax2.set_xlabel('F1-Score')
        ax2.set_title('Model F1-Score Karşılaştırması')
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'model_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # CSV olarak kaydet
        comparison_df.to_csv(os.path.join(self.results_dir, 'model_comparison.csv'), index=False)
        print(f"\n✓ Karşılaştırma sonuçları kaydedildi: results/model_comparison.csv")
        
        return comparison_df
    
    def save_models(self):
        """
        Modelleri kaydet
        """
        models_dir = os.path.join(self.results_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            if 'LSTM' in model_name:
                # Keras modelleri için
                model_path = os.path.join(models_dir, f'{model_name.replace(" ", "_")}.h5')
                model.save(model_path)
            else:
                # Scikit-learn modelleri için
                model_path = os.path.join(models_dir, f'{model_name.replace(" ", "_")}.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            print(f"✓ {model_name} modeli kaydedildi: {model_path}")
    
    def get_training_histories(self):
        """
        Eğitim geçmişlerini döndür
        """
        histories = {}
        for model_name, results in self.results.items():
            if 'history' in results:
                histories[model_name] = results['history']
            else:
                histories[model_name] = None
        return histories


def main():
    """
    Ana işlem fonksiyonu
    """
    # Veri ön işleme sonuçlarını yükle
    from data_preprocessing import main as preprocess_main
    preprocess_results = preprocess_main()
    
    X_train = preprocess_results['X_train']
    X_val = preprocess_results['X_val']
    X_test = preprocess_results['X_test']
    y_train = preprocess_results['y_train']
    y_val = preprocess_results['y_val']
    y_test = preprocess_results['y_test']
    
    # Model trainer oluştur
    trainer = ModelTrainer()
    
    # Geleneksel modelleri eğit
    trainer.train_traditional_models(X_train, X_val, y_train, y_val)
    
    # Hiperparametre optimizasyonu
    trainer.hyperparameter_optimization(X_train, X_val, y_train, y_val)
    
    # LSTM için veriyi hazırla
    n_features = X_train.shape[1] if len(X_train.shape) > 1 else 1
    X_train_seq, y_train_cat, X_val_seq, y_val_cat, X_test_seq, y_test_cat, n_classes = \
        trainer.prepare_lstm_data(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # LSTM modeli eğit
    trainer.train_lstm_model(X_train_seq, X_val_seq, y_train_cat, y_val_cat, 
                            n_features, n_classes, epochs=50)
    
    # Bidirectional LSTM modeli eğit
    trainer.train_bidirectional_lstm_model(X_train_seq, X_val_seq, y_train_cat, y_val_cat,
                                          n_features, n_classes, epochs=50)
    
    # Modelleri karşılaştır
    comparison_df = trainer.compare_models()
    
    # Modelleri kaydet
    trainer.save_models()
    
    print("\n" + "=" * 60)
    print("MODELLEME TAMAMLANDI!")
    print("=" * 60)
    
    return trainer, comparison_df


if __name__ == "__main__":
    trainer, comparison = main()

