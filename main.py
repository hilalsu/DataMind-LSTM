"""
Ana Çalıştırma Scripti
Bu script tüm işlemleri sırayla çalıştırır.
"""

import os
import sys
import pandas as pd
import numpy as np

def main():
    """
    Tüm işlemleri sırayla çalıştır
    """
    print("=" * 60)
    print("DATAMIND-LSTM PROJESİ")
    print("=" * 60)
    print("\nBu script veri ön işleme ve modelleme işlemlerini çalıştırır.\n")
    
    # 1. Veri ön işleme
    print("\n" + "=" * 60)
    print("ADIM 1: VERİ ÖN İŞLEME")
    print("=" * 60)
    try:
        from data_preprocessing import main as preprocess_main
        preprocess_results = preprocess_main()
        print("\n✓ Veri ön işleme başarıyla tamamlandı!")
    except Exception as e:
        print(f"\n✗ Veri ön işleme sırasında hata oluştu: {e}")
        sys.exit(1)
    
    # 2. Modelleme
    print("\n" + "=" * 60)
    print("ADIM 2: MODELLEME")
    print("=" * 60)
    try:
        from modeling import ModelTrainer
        import pandas as pd
        import numpy as np
        
        # Veri ön işleme sonuçlarını kullan
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
        
        print("\n✓ Modelleme başarıyla tamamlandı!")
        
        # 3. Değerlendirme
        print("\n" + "=" * 60)
        print("ADIM 3: DEĞERLENDİRME")
        print("=" * 60)
        
        from evaluation import ModelEvaluator
        
        evaluator = ModelEvaluator()
        
        # Tüm modelleri değerlendir
        all_models = trainer.models
        
        # Geleneksel modeller için normal değerlendirme
        traditional_models = {k: v for k, v in all_models.items() if 'LSTM' not in k}
        lstm_models = {k: v for k, v in all_models.items() if 'LSTM' in k}
        
        # Geleneksel modelleri değerlendir
        if traditional_models:
            metrics_df_trad = evaluator.evaluate_all_models(
                traditional_models, X_val, y_val, X_test, y_test
            )
        
        # LSTM modellerini değerlendir (sequence verisi ile)
        if lstm_models:
            # LSTM modelleri için özel değerlendirme
            for model_name, model in lstm_models.items():
                try:
                    y_pred_proba = model.predict(X_val_seq, verbose=0)
                    y_pred = np.argmax(y_pred_proba, axis=1)
                    y_true = np.argmax(y_val_cat, axis=1)
                    
                    # Metrikleri hesapla
                    metrics = evaluator.calculate_metrics(y_true, y_pred, model_name)
                    
                    # Confusion matrix
                    evaluator.plot_confusion_matrix(y_true, y_pred, model_name)
                    
                    print(f"✓ {model_name} değerlendirildi")
                except Exception as e:
                    print(f"  ⚠ {model_name} değerlendirme hatası: {e}")
        
        # Tüm modeller için birleşik metrik tablosu
        all_metrics = []
        for model_name in all_models.keys():
            if model_name in evaluator.evaluation_results:
                metrics = evaluator.evaluation_results[model_name]
                all_metrics.append({
                    'Model': model_name,
                    'Accuracy': metrics['accuracy'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1_score']
                })
        
        if all_metrics:
            metrics_df = pd.DataFrame(all_metrics)
            metrics_df = metrics_df.sort_values('F1-Score', ascending=False)
            csv_path = os.path.join(evaluator.results_dir, 'evaluation_metrics.csv')
            metrics_df.to_csv(csv_path, index=False)
            evaluator.plot_metrics_comparison(metrics_df)
            print(f"✓ Tüm metrikler kaydedildi: {csv_path}")
        else:
            metrics_df = pd.DataFrame()
        
        # Overfitting analizi
        training_histories = trainer.get_training_histories()
        overfitting_analysis = evaluator.analyze_overfitting(all_models, training_histories)
        
        # En iyi modeli seç
        if not metrics_df.empty:
            best_model_name, best_model, model_info = evaluator.select_best_model(
                metrics_df, all_models
            )
        else:
            print("⚠ Metrikler bulunamadı, en iyi model seçilemedi")
            best_model_name = None
        
        print("\n✓ Değerlendirme başarıyla tamamlandı!")
        
        # 4. Rapor oluştur
        print("\n" + "=" * 60)
        print("ADIM 4: RAPOR OLUŞTURMA")
        print("=" * 60)
        
        try:
            from reporting import ReportGenerator
            report_generator = ReportGenerator()
            report_path = report_generator.generate_report()
            print(f"\n✓ Rapor başarıyla oluşturuldu: {report_path}")
        except Exception as e:
            print(f"⚠ Rapor oluşturma hatası: {e}")
        
    except Exception as e:
        print(f"\n✗ Modelleme sırasında hata oluştu: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 4. Özet
    print("\n" + "=" * 60)
    print("TÜM İŞLEMLER TAMAMLANDI!")
    print("=" * 60)
    print("\nSonuçlar 'results/' klasöründe bulunabilir:")
    print("  - Görselleştirmeler: *.png dosyaları")
    print("  - İşlenmiş veri: processed_data.csv")
    print("  - Model karşılaştırması: model_comparison.csv")
    print("  - Değerlendirme metrikleri: evaluation_metrics.csv")
    print("  - Confusion matrix'ler: confusion_matrix_*.png")
    print("  - Overfitting analizi: overfitting_analysis.json")
    print("  - Eğitilmiş modeller: models/ klasörü")
    print("  - En iyi model: models/best_model.*")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

