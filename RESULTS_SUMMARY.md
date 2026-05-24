# 📊 Model Eğitim ve Karşılaştırma Sonuçları

Makine öğrenmesi ve derin öğrenme modellerimizin futbolcu piyasa değeri tahmin veri seti (yaklaşık 55 bin eğitim, 13 bin test verisi) üzerindeki performans karşılaştırma sonuçları aşağıda sunulmuştur.

## Performans Tablosu

| Model | R2 Skoru (Hata Açıklama) | RMSE (Hata) | MAE (Hata) | Eğitim Süresi |
| :--- | :--- | :--- | :--- | :--- |
| **LightGBM** | **%82.10** (0.8210) | **1.628.906** | 351.489 | **3.95 sn** |
| **CatBoost** | %80.55 (0.8055) | 1.697.661 | 368.051 | 4.49 sn |
| **Random Forest** | %80.19 (0.8019) | 1.713.539 | **323.614** | 52.86 sn |
| **XGBoost** | %77.63 (0.7763) | 1.820.758 | 331.593 | 4.12 sn |
| **Ridge Regression** | %63.41 (0.6341) | 2.328.664 | 729.421 | 0.04 sn |
| **TabNet** | %27.44 (0.2744) | 3.279.345 | 739.573 | 599.83 sn |

*(Not: R2 Skorunun 1'e yani %100'e yakın olması, RMSE ve MAE değerlerinin ise düşük olması daha iyidir.)*

## Sonuçların Analizi

1. **En Başarılı Model:** Gerek yüksek doğruluk oranı (%82.10 R2) gerekse inanılmaz hızlı eğitim süresi (yaklaşık 4 saniye) ile **LightGBM** bu veri seti için açık ara en iyi tercih olarak öne çıkmaktadır.
2. **Ortalama Mutlak Hata (MAE):** MAE bakımından en düşük hatayı veren model **Random Forest** (ortalama 323.000 €'luk sapma) olmuştur. Ancak eğitim süresi diğer ağaç bazlı modellere kıyasla daha uzun sürmektedir (53 saniye).

## Çıktılar

Eğitilen tüm modeller, bir sonraki kullanımınız (inference) için otomatik olarak `results/saved_models/` klasörüne kaydedilmiştir. 
Bununla birlikte elde edilen metriklerin raw hali `results/benchmark_results.csv` dosyasında, görselleştirilmiş hali ise `results/benchmark_plot.png` grafik dosyasında mevcuttur.
