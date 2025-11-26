"""
Raporlama Modülü
Detaylı analiz raporu oluşturur
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from pathlib import Path

class ReportGenerator:
    """
    Rapor oluşturucu sınıfı
    """
    
    def __init__(self, results_dir='results'):
        self.results_dir = results_dir
        self.report_path = os.path.join(results_dir, 'project_report.html')
    
    def generate_report(self):
        """
        Detaylı rapor oluştur
        """
        print("\n" + "=" * 60)
        print("RAPOR OLUŞTURULUYOR")
        print("=" * 60)
        
        html_content = self._create_html_template()
        
        # Raporu kaydet
        with open(self.report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✓ Rapor kaydedildi: {self.report_path}")
        return self.report_path
    
    def _create_html_template(self):
        """
        HTML rapor şablonu oluştur
        """
        # Veri toplama
        data_info = self._get_data_info()
        preprocessing_info = self._get_preprocessing_info()
        model_info = self._get_model_info()
        evaluation_info = self._get_evaluation_info()
        overfitting_info = self._get_overfitting_info()
        
        html = f"""
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DataMind-LSTM Proje Raporu</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #1f77b4;
            border-bottom: 3px solid #1f77b4;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #2c3e50;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 10px;
        }}
        h3 {{
            color: #34495e;
            margin-top: 20px;
        }}
        .section {{
            margin: 20px 0;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 5px;
        }}
        .metric {{
            display: inline-block;
            margin: 10px;
            padding: 15px;
            background-color: #ecf0f1;
            border-radius: 5px;
            min-width: 150px;
        }}
        .metric-label {{
            font-weight: bold;
            color: #7f8c8d;
        }}
        .metric-value {{
            font-size: 24px;
            color: #2c3e50;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .success {{
            color: #27ae60;
            font-weight: bold;
        }}
        .warning {{
            color: #f39c12;
            font-weight: bold;
        }}
        .error {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .code-block {{
            background-color: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
            text-align: center;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 DataMind-LSTM Proje Raporu</h1>
        <p><strong>Oluşturulma Tarihi:</strong> {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
        
        <h2>1. Proje Özeti</h2>
        <div class="section">
            <p>Bu proje, kadın cinayetleri verilerini analiz etmek ve farklı makine öğrenmesi 
            modelleri kullanarak tahmin yapmak amacıyla geliştirilmiştir.</p>
            <p><strong>Hedef:</strong> Fail durumunu (killer_status) tahmin etmek</p>
        </div>
        
        <h2>2. Veri Kaynakları</h2>
        <div class="section">
            {data_info}
        </div>
        
        <h2>3. Uygulanan İşlemler</h2>
        <div class="section">
            {preprocessing_info}
        </div>
        
        <h2>4. Model Karşılaştırmaları</h2>
        <div class="section">
            {model_info}
        </div>
        
        <h2>5. Değerlendirme Metrikleri</h2>
        <div class="section">
            {evaluation_info}
        </div>
        
        <h2>6. Overfitting Analizi</h2>
        <div class="section">
            {overfitting_info}
        </div>
        
        <h2>7. Model Başarısı ve Yorumlar</h2>
        <div class="section">
            {self._get_interpretation()}
        </div>
        
        <h2>8. Sonuçlar ve Öneriler</h2>
        <div class="section">
            {self._get_conclusions()}
        </div>
        
        <div class="footer">
            <p>DataMind-LSTM Projesi | {datetime.now().year}</p>
        </div>
    </div>
</body>
</html>
"""
        return html
    
    def _get_data_info(self):
        """
        Veri kaynağı bilgileri
        """
        dataset_dir = Path("dataset")
        files = list(dataset_dir.glob("*.csv")) if dataset_dir.exists() else []
        
        info = f"""
        <h3>Veri Dosyaları</h3>
        <ul>
            <li><strong>Toplam Dosya Sayısı:</strong> {len(files)}</li>
        </ul>
        """
        
        if files:
            info += "<ul>"
            for file in files:
                try:
                    df = pd.read_csv(file, encoding='utf-8')
                    info += f"<li><strong>{file.name}</strong>: {len(df)} satır, {len(df.columns)} kolon</li>"
                except:
                    info += f"<li><strong>{file.name}</strong>: Yüklenemedi</li>"
            info += "</ul>"
        
        if os.path.exists("results/processed_data.csv"):
            df_processed = pd.read_csv("results/processed_data.csv")
            info += f"""
            <h3>İşlenmiş Veri</h3>
            <ul>
                <li><strong>Toplam Kayıt:</strong> {len(df_processed)}</li>
                <li><strong>Özellik Sayısı:</strong> {len(df_processed.columns)}</li>
            </ul>
            """
        
        return info
    
    def _get_preprocessing_info(self):
        """
        Ön işleme bilgileri
        """
        info = """
        <h3>Uygulanan İşlemler</h3>
        <ol>
            <li><strong>Veri Yükleme ve Birleştirme:</strong> İki farklı CSV dosyası birleştirildi</li>
            <li><strong>Veri Temizleme:</strong>
                <ul>
                    <li>Eksik veri analizi ve işleme</li>
                    <li>Boşluk temizleme</li>
                    <li>Tarih formatı dönüşümü</li>
                    <li>Yaş verilerinin sayısal forma dönüştürülmesi</li>
                    <li>Aykırı değer tespiti ve işleme (IQR yöntemi)</li>
                    <li>Tekrarlanan satırların kaldırılması</li>
                </ul>
            </li>
            <li><strong>Encoding:</strong> Label Encoding ile kategorik değişkenler sayısal forma dönüştürüldü</li>
            <li><strong>Normalizasyon:</strong> StandardScaler ile sayısal değişkenler normalize edildi</li>
            <li><strong>Korelasyon Analizi:</strong> Değişkenler arası ilişkiler analiz edildi</li>
            <li><strong>Veri Bölme:</strong> Train (%70), Validation (%10), Test (%20) olarak bölündü</li>
        </ol>
        """
        return info
    
    def _get_model_info(self):
        """
        Model bilgileri
        """
        info = "<h3>Eğitilen Modeller</h3>"
        
        if os.path.exists("results/model_comparison.csv"):
            df = pd.read_csv("results/model_comparison.csv")
            info += "<table>"
            info += "<tr><th>Model</th><th>Accuracy</th><th>F1-Score</th></tr>"
            for _, row in df.iterrows():
                info += f"<tr><td>{row['Model']}</td><td>{row['Accuracy']:.4f}</td><td>{row['F1-Score']:.4f}</td></tr>"
            info += "</table>"
        else:
            info += "<p>Model karşılaştırması bulunamadı</p>"
        
        info += """
        <h3>Model Detayları</h3>
        <ul>
            <li><strong>Logistic Regression:</strong> Doğrusal sınıflandırma modeli</li>
            <li><strong>Random Forest:</strong> Ensemble yöntemi, 100 ağaç</li>
            <li><strong>SVM:</strong> RBF kernel ile non-linear sınıflandırma</li>
            <li><strong>Random Forest (Optimized):</strong> RandomizedSearchCV ile optimize edilmiş</li>
            <li><strong>LSTM:</strong> 128-64 unit'li iki katmanlı LSTM</li>
            <li><strong>Bidirectional LSTM:</strong> İleri ve geri yönlü bilgi akışı</li>
        </ul>
        """
        
        return info
    
    def _get_evaluation_info(self):
        """
        Değerlendirme bilgileri
        """
        info = "<h3>Değerlendirme Metrikleri</h3>"
        
        if os.path.exists("results/evaluation_metrics.csv"):
            df = pd.read_csv("results/evaluation_metrics.csv")
            info += "<table>"
            info += "<tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-Score</th></tr>"
            for _, row in df.iterrows():
                info += f"""
                <tr>
                    <td>{row['Model']}</td>
                    <td>{row['Accuracy']:.4f}</td>
                    <td>{row['Precision']:.4f}</td>
                    <td>{row['Recall']:.4f}</td>
                    <td>{row['F1-Score']:.4f}</td>
                </tr>
                """
            info += "</table>"
        else:
            info += "<p>Değerlendirme metrikleri bulunamadı</p>"
        
        # En iyi model
        if os.path.exists("results/models/best_model_info.json"):
            with open("results/models/best_model_info.json", 'r', encoding='utf-8') as f:
                best_model = json.load(f)
            info += f"""
            <h3>🏆 En İyi Model</h3>
            <div class="metric">
                <div class="metric-label">Model</div>
                <div class="metric-value">{best_model['model_name']}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Accuracy</div>
                <div class="metric-value">{best_model['accuracy']:.4f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">F1-Score</div>
                <div class="metric-value">{best_model['f1_score']:.4f}</div>
            </div>
            """
        
        return info
    
    def _get_overfitting_info(self):
        """
        Overfitting analizi bilgileri
        """
        info = "<h3>Overfitting Analizi Sonuçları</h3>"
        
        if os.path.exists("results/overfitting_analysis.json"):
            with open("results/overfitting_analysis.json", 'r', encoding='utf-8') as f:
                overfitting_data = json.load(f)
            
            info += "<table>"
            info += "<tr><th>Model</th><th>Train Acc</th><th>Val Acc</th><th>Gap</th><th>Risk</th></tr>"
            
            for model_name, data in overfitting_data.items():
                train_acc = data.get('train_accuracy', 'N/A')
                val_acc = data.get('val_accuracy', 'N/A')
                gap = data.get('accuracy_gap', 'N/A')
                risk = data.get('overfitting_risk', 'N/A')
                
                if isinstance(gap, (int, float)):
                    gap_str = f"{gap:.4f}"
                    risk_class = "error" if gap > 0.1 else "success" if gap < 0.05 else "warning"
                else:
                    gap_str = str(gap)
                    risk_class = ""
                
                if isinstance(train_acc, (int, float)):
                    train_acc_str = f"{train_acc:.4f}"
                else:
                    train_acc_str = str(train_acc)
                
                if isinstance(val_acc, (int, float)):
                    val_acc_str = f"{val_acc:.4f}"
                else:
                    val_acc_str = str(val_acc)
                
                info += f"""
                <tr>
                    <td>{model_name}</td>
                    <td>{train_acc_str}</td>
                    <td>{val_acc_str}</td>
                    <td>{gap_str}</td>
                    <td class="{risk_class}">{risk}</td>
                </tr>
                """
            info += "</table>"
            
            info += """
            <h3>Overfitting Yorumu</h3>
            <ul>
                <li><strong>Düşük Risk (&lt;0.05):</strong> Model genelleme yapıyor, overfitting yok</li>
                <li><strong>Orta Risk (0.05-0.1):</strong> Hafif overfitting var, dikkat edilmeli</li>
                <li><strong>Yüksek Risk (&gt;0.1):</strong> Ciddi overfitting var, model düzenlenmeli</li>
            </ul>
            """
        else:
            info += "<p>Overfitting analizi bulunamadı</p>"
        
        return info
    
    def _get_interpretation(self):
        """
        Model başarısı yorumları
        """
        interpretation = """
        <h3>Model Performans Değerlendirmesi</h3>
        <p>Modellerin performansı aşağıdaki metriklerle değerlendirilmiştir:</p>
        <ul>
            <li><strong>Accuracy:</strong> Genel doğruluk oranı</li>
            <li><strong>Precision:</strong> Pozitif tahminlerin doğruluğu</li>
            <li><strong>Recall:</strong> Gerçek pozitiflerin yakalanma oranı</li>
            <li><strong>F1-Score:</strong> Precision ve Recall'un harmonik ortalaması</li>
        </ul>
        
        <h3>Model Başarı Faktörleri</h3>
        <ul>
            <li><strong>Veri Kalitesi:</strong> Temizlenmiş ve normalize edilmiş veri</li>
            <li><strong>Özellik Mühendisliği:</strong> Encoding ve normalizasyon işlemleri</li>
            <li><strong>Model Seçimi:</strong> Farklı algoritmaların denenmesi</li>
            <li><strong>Hiperparametre Optimizasyonu:</strong> En iyi parametrelerin bulunması</li>
        </ul>
        
        <h3>Zorluklar ve Sınırlamalar</h3>
        <ul>
            <li>Eksik veri oranının yüksek olması</li>
            <li>Dengesiz sınıf dağılımı</li>
            <li>Kategorik değişkenlerin çokluğu</li>
            <li>Zaman serisi verisi olmayan tabular veri için LSTM kullanımı</li>
        </ul>
        """
        return interpretation
    
    def _get_conclusions(self):
        """
        Sonuçlar ve öneriler
        """
        conclusions = """
        <h3>Ana Bulgular</h3>
        <ul>
            <li>Farklı model türleri denenmiş ve karşılaştırılmıştır</li>
            <li>En iyi performans gösteren model belirlenmiştir</li>
            <li>Overfitting analizi yapılmış ve risk değerlendirilmiştir</li>
            <li>Detaylı değerlendirme metrikleri hesaplanmıştır</li>
        </ul>
        
        <h3>Öneriler</h3>
        <ul>
            <li>Daha fazla veri toplanması</li>
            <li>Özellik mühendisliği ile yeni özellikler eklenmesi</li>
            <li>Ensemble yöntemlerinin denenmesi</li>
            <li>Cross-validation ile daha güvenilir sonuçlar elde edilmesi</li>
            <li>Model interpretability için SHAP veya LIME kullanılması</li>
        </ul>
        
        <h3>Gelecek Çalışmalar</h3>
        <ul>
            <li>Daha fazla veri ile model performansının artırılması</li>
            <li>Transfer learning yaklaşımlarının denenmesi</li>
            <li>Real-time tahmin sistemi geliştirilmesi</li>
            <li>Model deployment ve API geliştirilmesi</li>
        </ul>
        """
        return conclusions


def main():
    """
    Ana fonksiyon
    """
    generator = ReportGenerator()
    report_path = generator.generate_report()
    print(f"\n✓ Rapor başarıyla oluşturuldu: {report_path}")
    return report_path


if __name__ == "__main__":
    main()

