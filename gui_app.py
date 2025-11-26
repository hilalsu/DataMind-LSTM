"""
GUI Uygulaması - Streamlit
Kullanıcı arayüzü ile veri yükleme, model çalıştırma ve sonuç görüntüleme
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Sayfa yapılandırması
st.set_page_config(
    page_title="DataMind-LSTM Projesi",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS stilleri
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

class GUIApp:
    """
    GUI Uygulama sınıfı
    """
    
    def __init__(self):
        self.data = None
        self.processed_data = None
        self.models = {}
        self.results = {}
        
    def run(self):
        """
        Ana uygulama
        """
        # Başlık
        st.markdown('<h1 class="main-header">📊 DataMind-LSTM Projesi</h1>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Sidebar menü
        st.sidebar.title("📋 Menü")
        page = st.sidebar.radio(
            "Sayfa Seçin:",
            ["🏠 Ana Sayfa", "📁 Veri Yükleme", "🔧 Veri Ön İşleme", 
             "🤖 Model Eğitimi", "📊 Sonuçlar ve Değerlendirme", "📈 Grafikler"]
        )
        
        if page == "🏠 Ana Sayfa":
            self.show_home()
        elif page == "📁 Veri Yükleme":
            self.show_data_loading()
        elif page == "🔧 Veri Ön İşleme":
            self.show_preprocessing()
        elif page == "🤖 Model Eğitimi":
            self.show_model_training()
        elif page == "📊 Sonuçlar ve Değerlendirme":
            self.show_results()
        elif page == "📈 Grafikler":
            self.show_visualizations()
    
    def show_home(self):
        """
        Ana sayfa
        """
        st.header("🏠 Ana Sayfa")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Veri Seti", "2 CSV Dosyası", "dataset/")
        
        with col2:
            results_count = len([f for f in os.listdir('results') if f.endswith('.png')]) if os.path.exists('results') else 0
            st.metric("Görselleştirme", f"{results_count} Grafik", "results/")
        
        with col3:
            models_count = len([f for f in os.listdir('results/models') if os.path.exists('results/models')]) if os.path.exists('results/models') else 0
            st.metric("Modeller", f"{models_count} Model", "models/")
        
        st.markdown("---")
        
        st.subheader("📖 Proje Hakkında")
        st.write("""
        Bu proje, kadın cinayetleri verilerini analiz etmek ve farklı makine öğrenmesi 
        modelleri kullanarak tahmin yapmak amacıyla geliştirilmiştir.
        
        **Özellikler:**
        - ✅ Veri ön işleme ve temizleme
        - ✅ Görselleştirme ve analiz
        - ✅ Geleneksel ML modelleri (Logistic Regression, Random Forest, SVM)
        - ✅ Derin öğrenme modelleri (LSTM, Bidirectional LSTM)
        - ✅ Model değerlendirme ve karşılaştırma
        - ✅ Overfitting analizi
        """)
        
        st.subheader("🚀 Hızlı Başlangıç")
        st.write("""
        1. **Veri Yükleme**: Sol menüden "Veri Yükleme" sayfasına gidin
        2. **Veri Ön İşleme**: Verileri temizleyin ve hazırlayın
        3. **Model Eğitimi**: Modelleri eğitin
        4. **Sonuçlar**: Sonuçları görüntüleyin ve analiz edin
        """)
    
    def show_data_loading(self):
        """
        Veri yükleme sayfası
        """
        st.header("📁 Veri Yükleme")
        
        st.subheader("Mevcut Veri Dosyaları")
        
        dataset_dir = Path("dataset")
        if dataset_dir.exists():
            files = list(dataset_dir.glob("*.csv"))
            if files:
                st.success(f"✓ {len(files)} CSV dosyası bulundu")
                for file in files:
                    st.write(f"  - {file.name}")
            else:
                st.warning("⚠ Veri dosyası bulunamadı")
        else:
            st.error("❌ dataset/ klasörü bulunamadı")
        
        st.markdown("---")
        
        st.subheader("Yeni Veri Yükle")
        uploaded_file = st.file_uploader(
            "CSV dosyası seçin",
            type=['csv'],
            help="Yeni bir veri dosyası yükleyebilirsiniz"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
                st.success(f"✓ Veri başarıyla yüklendi: {len(df)} satır, {len(df.columns)} kolon")
                
                st.subheader("Veri Önizleme")
                st.dataframe(df.head(10))
                
                st.subheader("Veri İstatistikleri")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Temel Bilgiler:**")
                    st.write(f"- Satır sayısı: {len(df)}")
                    st.write(f"- Kolon sayısı: {len(df.columns)}")
                    st.write(f"- Eksik değer: {df.isnull().sum().sum()}")
                
                with col2:
                    st.write("**Veri Tipleri:**")
                    dtype_counts = df.dtypes.value_counts()
                    for dtype, count in dtype_counts.items():
                        st.write(f"- {dtype}: {count}")
                
                # Veriyi session state'e kaydet
                st.session_state['uploaded_data'] = df
                
            except Exception as e:
                st.error(f"❌ Hata: {e}")
    
    def show_preprocessing(self):
        """
        Veri ön işleme sayfası
        """
        st.header("🔧 Veri Ön İşleme")
        
        if st.button("🔄 Veri Ön İşlemeyi Çalıştır", type="primary"):
            with st.spinner("Veri ön işleme yapılıyor..."):
                try:
                    from data_preprocessing import main as preprocess_main
                    results = preprocess_main()
                    
                    st.session_state['preprocessing_results'] = results
                    st.success("✓ Veri ön işleme tamamlandı!")
                    
                    # Sonuçları göster
                    st.subheader("İşleme Sonuçları")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Train Seti", f"{len(results['X_train'])} örnek")
                    with col2:
                        st.metric("Validation Seti", f"{len(results['X_val'])} örnek")
                    with col3:
                        st.metric("Test Seti", f"{len(results['X_test'])} örnek")
                    
                except Exception as e:
                    st.error(f"❌ Hata: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        
        # İşlenmiş veriyi göster
        if os.path.exists("results/processed_data.csv"):
            st.markdown("---")
            st.subheader("İşlenmiş Veri")
            if st.button("📊 İşlenmiş Veriyi Göster"):
                df_processed = pd.read_csv("results/processed_data.csv")
                st.dataframe(df_processed.head(20))
                st.write(f"Toplam: {len(df_processed)} satır")
    
    def show_model_training(self):
        """
        Model eğitimi sayfası
        """
        st.header("🤖 Model Eğitimi")
        
        st.subheader("Model Seçimi")
        
        model_options = {
            "Tüm Modeller": "all",
            "Geleneksel ML Modelleri": "traditional",
            "Derin Öğrenme Modelleri": "deep_learning",
            "Sadece LSTM": "lstm",
            "Sadece Bidirectional LSTM": "bilstm"
        }
        
        selected_models = st.multiselect(
            "Eğitilecek modelleri seçin:",
            list(model_options.keys()),
            default=["Tüm Modeller"]
        )
        
        epochs = st.slider("Epoch sayısı (LSTM için):", 10, 100, 50)
        
        if st.button("🚀 Modelleri Eğit", type="primary"):
            with st.spinner("Modeller eğitiliyor... Bu işlem biraz zaman alabilir."):
                try:
                    # Veri ön işleme sonuçlarını kontrol et
                    if 'preprocessing_results' not in st.session_state:
                        st.warning("⚠ Önce veri ön işleme yapılmalı!")
                        return
                    
                    results = st.session_state['preprocessing_results']
                    X_train = results['X_train']
                    X_val = results['X_val']
                    X_test = results['X_test']
                    y_train = results['y_train']
                    y_val = results['y_val']
                    y_test = results['y_test']
                    
                    from modeling import ModelTrainer
                    trainer = ModelTrainer()
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Geleneksel modeller
                    if "all" in selected_models or "traditional" in selected_models:
                        status_text.text("Geleneksel modeller eğitiliyor...")
                        trainer.train_traditional_models(X_train, X_val, y_train, y_val)
                        progress_bar.progress(30)
                        
                        status_text.text("Hiperparametre optimizasyonu yapılıyor...")
                        trainer.hyperparameter_optimization(X_train, X_val, y_train, y_val)
                        progress_bar.progress(50)
                    
                    # LSTM modelleri
                    if "all" in selected_models or "deep_learning" in selected_models or "lstm" in selected_models or "bilstm" in selected_models:
                        status_text.text("LSTM verileri hazırlanıyor...")
                        n_features = X_train.shape[1] if len(X_train.shape) > 1 else 1
                        X_train_seq, y_train_cat, X_val_seq, y_val_cat, X_test_seq, y_test_cat, n_classes = \
                            trainer.prepare_lstm_data(X_train, X_val, X_test, y_train, y_val, y_test)
                        progress_bar.progress(60)
                        
                        if "all" in selected_models or "deep_learning" in selected_models or "lstm" in selected_models:
                            status_text.text("LSTM modeli eğitiliyor...")
                            trainer.train_lstm_model(X_train_seq, X_val_seq, y_train_cat, y_val_cat, 
                                                    n_features, n_classes, epochs=epochs)
                            progress_bar.progress(80)
                        
                        if "all" in selected_models or "deep_learning" in selected_models or "bilstm" in selected_models:
                            status_text.text("Bidirectional LSTM modeli eğitiliyor...")
                            trainer.train_bidirectional_lstm_model(X_train_seq, X_val_seq, y_train_cat, y_val_cat,
                                                                  n_features, n_classes, epochs=epochs)
                            progress_bar.progress(90)
                    
                    # Modelleri karşılaştır
                    status_text.text("Modeller karşılaştırılıyor...")
                    comparison_df = trainer.compare_models()
                    trainer.save_models()
                    progress_bar.progress(100)
                    
                    st.session_state['trainer'] = trainer
                    st.session_state['comparison_df'] = comparison_df
                    
                    st.success("✓ Model eğitimi tamamlandı!")
                    status_text.empty()
                    progress_bar.empty()
                    
                except Exception as e:
                    st.error(f"❌ Hata: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        
        # Eğitilmiş modelleri göster
        if os.path.exists("results/models"):
            st.markdown("---")
            st.subheader("Eğitilmiş Modeller")
            model_files = [f for f in os.listdir("results/models") if f.endswith(('.pkl', '.h5'))]
            if model_files:
                st.write(f"✓ {len(model_files)} model bulundu:")
                for model_file in model_files:
                    st.write(f"  - {model_file}")
            else:
                st.info("Henüz model eğitilmemiş")
    
    def show_results(self):
        """
        Sonuçlar ve değerlendirme sayfası
        """
        st.header("📊 Sonuçlar ve Değerlendirme")
        
        # Model karşılaştırması
        if os.path.exists("results/model_comparison.csv"):
            st.subheader("Model Karşılaştırması")
            comparison_df = pd.read_csv("results/model_comparison.csv")
            st.dataframe(comparison_df, use_container_width=True)
            
            # Görselleştirme
            if os.path.exists("results/model_comparison.png"):
                st.image("results/model_comparison.png", use_container_width=True)
        
        # Değerlendirme metrikleri
        if os.path.exists("results/evaluation_metrics.csv"):
            st.subheader("Değerlendirme Metrikleri")
            metrics_df = pd.read_csv("results/evaluation_metrics.csv")
            st.dataframe(metrics_df, use_container_width=True)
            
            # Metrik grafikleri
            if os.path.exists("results/metrics_comparison.png"):
                st.image("results/metrics_comparison.png", use_container_width=True)
        
        # Confusion matrix'ler
        st.subheader("Confusion Matrix'ler")
        confusion_files = [f for f in os.listdir("results") if f.startswith("confusion_matrix_")]
        if confusion_files:
            cols = st.columns(min(2, len(confusion_files)))
            for idx, file in enumerate(confusion_files[:4]):  # İlk 4'ü göster
                with cols[idx % 2]:
                    st.image(f"results/{file}", caption=file.replace("confusion_matrix_", "").replace(".png", ""))
        else:
            st.info("Confusion matrix bulunamadı")
        
        # Overfitting analizi
        if os.path.exists("results/overfitting_analysis.json"):
            st.subheader("Overfitting Analizi")
            import json
            with open("results/overfitting_analysis.json", 'r', encoding='utf-8') as f:
                overfitting_data = json.load(f)
            
            for model_name, data in overfitting_data.items():
                with st.expander(f"📈 {model_name}"):
                    if data.get('train_accuracy'):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Train Accuracy", f"{data['train_accuracy']:.4f}")
                        with col2:
                            st.metric("Val Accuracy", f"{data['val_accuracy']:.4f}")
                        with col3:
                            gap = data.get('accuracy_gap', 0)
                            st.metric("Accuracy Gap", f"{gap:.4f}", 
                                     delta="Overfitting" if gap > 0.1 else "Normal")
                        
                        st.write(f"**Overfitting Risk:** {data.get('overfitting_risk', 'N/A')}")
            
            # Overfitting grafikleri
            overfitting_files = [f for f in os.listdir("results") if f.startswith("overfitting_analysis_")]
            if overfitting_files:
                for file in overfitting_files[:2]:  # İlk 2'yi göster
                    st.image(f"results/{file}", use_container_width=True)
        
        # En iyi model
        if os.path.exists("results/models/best_model_info.json"):
            st.subheader("🏆 En İyi Model")
            import json
            with open("results/models/best_model_info.json", 'r', encoding='utf-8') as f:
                best_model_info = json.load(f)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Model", best_model_info['model_name'])
            with col2:
                st.metric("Accuracy", f"{best_model_info['accuracy']:.4f}")
            with col3:
                st.metric("F1-Score", f"{best_model_info['f1_score']:.4f}")
            with col4:
                st.metric("Precision", f"{best_model_info['precision']:.4f}")
    
    def show_visualizations(self):
        """
        Grafikler sayfası
        """
        st.header("📈 Grafikler")
        
        # Mevcut grafikleri listele
        if os.path.exists("results"):
            graph_files = [f for f in os.listdir("results") if f.endswith('.png')]
            
            if graph_files:
                st.subheader("Mevcut Grafikler")
                
                # Kategorilere ayır
                categories = {
                    "Veri Dağılımları": [f for f in graph_files if any(x in f for x in ['distribution', 'boxplot'])],
                    "Korelasyon": [f for f in graph_files if 'correlation' in f],
                    "Model Karşılaştırmaları": [f for f in graph_files if 'comparison' in f],
                    "Confusion Matrix": [f for f in graph_files if 'confusion' in f],
                    "Eğitim Geçmişi": [f for f in graph_files if 'training_history' in f or 'overfitting' in f],
                    "Diğer": [f for f in graph_files if not any(x in f for x in ['distribution', 'correlation', 'comparison', 'confusion', 'training', 'overfitting'])]
                }
                
                for category, files in categories.items():
                    if files:
                        with st.expander(f"📊 {category} ({len(files)})"):
                            cols = st.columns(min(2, len(files)))
                            for idx, file in enumerate(files):
                                with cols[idx % 2]:
                                    st.image(f"results/{file}", caption=file, use_container_width=True)
            else:
                st.info("Henüz grafik oluşturulmamış")
        else:
            st.warning("results/ klasörü bulunamadı")


def main():
    """
    Ana fonksiyon
    """
    app = GUIApp()
    app.run()


if __name__ == "__main__":
    main()

