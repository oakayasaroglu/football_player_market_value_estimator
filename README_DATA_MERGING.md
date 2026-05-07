# Data Merging İşlemleri ve Eksik Veri (NaN) Yönetimi

Bu doküman, `src/data_merging.py` betiği ile yapılan veri birleştirme işlemlerinde, eksik verilerin (`NaN`) nasıl ele alındığını açıklamaktadır. Sayısal ve mantıksal olarak **"hiç gerçekleşmemiş"** olaylar `0` ile doldurulurken, **"bilinmeyen"** veya **"zaman belirten"** veriler `NaN` (boş) bırakılmıştır.

## Değişken Gruplarına Göre Aksiyon Tablosu

| Veri Kategorisi / Sütun Adları | Boşsa Ne Oluyor? | Açıklama / Mantığı |
| :--- | :---: | :--- |
| **Profil Verileri** <br> `height`, `foot`, `joined`, `contract_expires`, `position`, `is_eu` vb. | **`NaN`** <br>*(Boş kalıyor)* | Bir oyuncunun boyu, sözleşme bitiş tarihi veya kullandığı ayak gibi temel veriler bilinmiyorsa buna 0 atanamaz. Bilgi gerçekte eksiktir ve `NaN` olarak korunur. |
| **Güncel Piyasa Değeri** <br> `value` (veya `latest_value` tablosundaki diğer sütunlar) | **`NaN`** <br>*(Boş kalıyor)* | Modelinizin muhtemel hedef değişkeni (target variable) olan market değeri bilinmiyorsa 0 yazılamaz (bu modeli yanıltır). Değerin tam olarak "bilinmediği" gösterilmek için `NaN` kalır. |
| **Kariyer İstatistikleri** <br> `career_goals`, `career_assists`, `career_minutes_played`, `career_yellow_cards` vb. (14 Sütun) | **`0`** | Performans tablosunda kaydı olmayan oyuncuların, kariyerleri boyunca bu kulvarda hiç oynamadığı, dolayısıyla gol/asist üretmediği varsayılır. Bu nedenle sıfırlanır. |
| **Son Dönem Performansları** <br> `recent_goals`, `recent_assists`, `recent_minutes_played`, `recent_nb_in_group`, `recent_yellow_cards` | **`0`** | Kariyer istatistikleriyle aynı mantıkta çalışır. Oyuncu son sezonlarda forma giymediyse hiç istatistik üretmemiş sayılır. |
| **Milli Takım Performansı** <br> `national_matches`, `national_goals` | **`0`** | Milli takım tablosunda kaydı bulunmayan oyuncunun, milli takıma hiç çağırılmadığı (0 maç, 0 gol) anlaşılır. |
| **Sakatlık (Sayısal Bilgiler)** <br> `injury_count`, `total_days_injured`, `total_games_missed` | **`0`** | Sakatlık geçmişi yoksa, bu durum oyuncunun hiç sakatlık yaşamadığını (0 kere sakatlandı, 0 gün ve maç kaçırdı) temsil eder. |
| **Sakatlık Tarihi** <br> `last_injury_date` | **`NaN`** <br>*(Boş kalıyor)* | Oyuncu hiç sakatlanmadıysa "son sakatlanma tarihi" diye bir şey olamaz. Tarih formatındaki bir veriye 0 atanamayacağı için `NaN` kalır. |
| **Transfer Geçmişi (Sayısal)** <br> `transfer_count`, `total_transfer_fees`, `max_transfer_fee`, `avg_transfer_fee`, `max_value_at_transfer`, `last_transfer_fee` | **`0`** | Transfer geçmişi yoksa, oyuncunun hiç transfer yapmadığı (0 kere transfer oldu, 0 € toplam bonservis ödendi) varsayılır. Bu kapsamda `last_transfer_fee` (son transfer ücreti) de dahil olmak üzere parasal ve sayısal değerler `0` olarak işlenir. |
| **Son Transfer Tarihi** <br> `last_transfer_date` | **`NaN`** <br>*(Boş kalıyor)* | Tarihte olduğu gibi, transferi olmayan birinin "son transfer tarihi" var olamayacağından boş kalır. |
