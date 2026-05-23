# Model Training Benchmark

Bu doküman, futbolcu piyasa değerlerini tahmin etmek için geliştirilen makine öğrenmesi model eğitim betiğinin (`model_training_benchmark.py`) detaylarını açıklar.

## 🚀 Kullanılan Modeller ve Altyapı
Uygulamamız makine öğrenmesi ve derin öğrenme tabanlı birden fazla regresyon algoritmasını karşılaştırmalı olarak test eder:
*   **Ridge Regression**: Temel (Baseline) doğrusal model
*   **Random Forest**: Ağaç tabanlı temel model
*   **XGBoost**: `tree_method='hist'` ve `device='cuda'` parametreleri ile GPU hızlandırması kullanır.
*   **LightGBM**: Hızlı ve hafif ağaç tabanlı model
*   **CatBoost**: `task_type='GPU'` parametresi ile GPU hızlandırması kullanır.

## 💾 Model Kaydetme Sistemi
Eğitim süresinden tasarruf etmek ve eğitilmiş modelleri canlı (inference) sistemlerde kullanabilmek için, her algoritma eğitim sonrasında otomatik olarak `results/saved_models/` dizini altına kaydedilir.

Formatlar şu şekildedir:
*   **Ridge, Random Forest, LightGBM:** `.pkl` (joblib kütüphanesi ile)
*   **XGBoost:** `.json` (kendi yerleşik save_model metodu ile)
*   **CatBoost:** `.cbm` (kendi yerleşik save_model metodu ile)

## 📊 Raporlama ve Çıktılar
Eğitimler bittikten sonra modeller şu metrikler üzerinden değerlendirilir:
*   **R2 Score:** Hata açıklama oranı (Yüksek olması daha iyidir).
*   **RMSE:** Kök Ortalama Kare Hatası (Düşük olması daha iyidir).
*   **MAE:** Ortalama Mutlak Hata (Düşük olması daha iyidir).
*   **Training Time:** Eğitim süresi (Düşük olması daha iyidir).

Bu sonuçlar `results/benchmark_results.csv` isimli bir tablo dosyasına yazılır ve `results/benchmark_plot.png` olarak üçlü grafik şeklinde görselleştirilir.

## 🛠️ Nasıl Çalıştırılır?
Model eğitim sürecini başlatmak için projenin ana dizininden şu komutu çalıştırabilirsiniz:

```bash
python src/model_training_benchmark.py
```
Eğitim işlemi bilgisayarınızın donanımına (özellikle ekran kartınıza) göre kısa süre içerisinde tamamlanacak ve kaydedilecektir.

## 🧠 Açıklanabilir Yapay Zeka (XAI)
Modellerin kararlarını anlamlandırmak ve hangi oyuncunun neden o piyasa değerine sahip olduğunu analiz etmek için projeye **SHAP** tabanlı XAI entegrasyonu yapılmıştır.
Analizler, benchmark sonuçlarında test setinde en yüksek başarıyı gösteren model üzerinden yapılmaktadır.

XAI analizini başlatmak ve açıklama grafiklerini (Özet, Özellik Önemi ve Şelale Grafikleri) oluşturmak için aşağıdaki komutu çalıştırabilirsiniz:

```bash
python src/xai_analysis.py
```
Oluşturulan grafikler `results/xai_plots/` dizinine kaydedilecektir.
