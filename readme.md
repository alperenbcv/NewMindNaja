# Doğrudan https://newmindnaja.streamlit.app/ adresine giderek projeye ulaşabilirsiniz.


## ÖZET
1-3300 adet eski hükümlüye ait mock data ile XGB modeli eğitilip bir pipeline kurulmuştur.

2-Bu modelin eğitimiyle model tekrar suç işleme riskini 0-1-2 (Low-Medium-High) şeklinde hesaplamaktadır. (recidivism)

3-Belli sayıda mahkeme kararı jsonl formatında oluşturulup, karar metinleri vektörize edilerek indexlenip neo4j'e kaydedilmiştir.

4-LLM entegre edilip, LLM'in doğru hareket edebilmesi için TOOL-AGENT'lar oluşturulmuştur.

5-Arayüz hazırlanıp streamlit üzerinden deploy'lanmıştır.

# PROJE DÖKÜMANTASYONU

## 1.  Amaç
Bu proje, yargılamaya yardımcı olmak üzere geliştirilen karar destek sistemi niteliğinde bir hukuk modülüdür. Esin kaynağı, Amerika Birleşik Devletleri'nde bazı eyaletlerde kullanılan ve hükümlülerin tekrar suç işleme (recidivism) riskini değerlendiren COMPAS (Correctional Offender Management Profiling for Alternative Sanctions) sistemidir. Ancak COMPAS’a yöneltilen başlıca eleştiriler; kararların şeffaf olmaması, değerlendirme ölçütlerinin açıklanmaması ve sistemin bir “kara kutu” olarak işlemesi yönündedir.
Geliştirilen bu proje, bu sorunlara çözüm olarak açıklanabilirliği, graf temelli görselleştirme ve LLM destekli örnek karar karşılaştırması ile sağlamayı hedefler. Sanığa ilişkin sosyo-demografik ve davranışsal veriler, graph database (Neo4j) üzerinde düğümler (nodes) ve ilişkiler (relations) yoluyla görselleştirilir. Aynı zamanda, semantic search destekli Retrieval-Augmented Generation (RAG) yaklaşımı ile benzer yargı kararları getirilebilir. Böylece sistem, sadece öngörücü bir model olmaktan çıkar, denetlenebilir, karşılaştırmalı ve norma duyarlı bir karar destek aracı haline gelir.
Sistem:
Sanığın recidivism riskini tahmin eder (ML tabanlı sınıflandırıcı),


Suç tipini, nitelikli halleri ve indirim sebeplerini değerlendirerek önerilen ceza miktarını hesaplar (TCK madde 61 ve 82'ye göre),


Hakimin verdiği cezayı analiz ederek, yasal normlara uygunluğunu değerlendirir,


LangChain ve Neo4j entegrasyonu sayesinde, semantik benzerliğe göre örnek kararları getirerek karar sürecine rehberlik eder.
## 2.  Teknik Mimarî
### 2.1. Kullanılan Teknolojiler
Python (veri işleme, ML modeli, arka plan işlemleri)


Streamlit (kullanıcı arayüzü)


XGBoost + SMOTE (risk tahmin modeli)


Neo4j (graph veritabanı, karar düğümleri)


LangChain (LLM koordinasyonu, agent & QA chain)


OpenAI Embedding API (kararların vektörleştirilmesi)


Scikit-learn + joblib (model pipeline & serializasyon)


Matplotlib (görselleştirme)

## 3.  Modelleme ve Açıklanabilirlik
### 3.1. Recidivism Modeli
Model, sahte ama gerçekçi şekilde üretilmiş sosyo-demografik ve davranışsal özellikleri kullanarak recidivism riskini 3 sınıfa ayırır:


🟢 Düşük Risk


🟡 Orta Risk


🔴 Yüksek Risk


Model çıktısı, SHAP değerleri ile desteklenebilir.


Tahmin güven düzeyleri, sınıflar arası olasılık dağılımı olarak gösterilir.


### 3.2. Cezanın Belirlenmesi
TCK 81-82 kapsamında suç tipi ve nitelikli hallere göre taban ceza belirlenir.


TCK 25–34 arası maddelere göre hafifletici / ortadan kaldırıcı sebepler değerlendirilir.


Sanığın motivasyonu ve risk seviyesi de göz önüne alınarak objektif bir ceza önerisi yapılır.


### 3.3. Hakim Cezası Analizi
Hakimin verdiği ceza ile model çıktısı karşılaştırılır.


Aşağıdaki türlerde yasal uyumsuzluklar tespit edilir:


Ağırlaştırıcı nitelik varken yetersiz ceza


Hafifletici neden varken ceza verilmesi


Recidivism düşük riskli olsa da cezanın aşırı olması


Sabit yıl cezası verilmesine rağmen suçun müebbetlik olması gibi


## 4. LLM Entegrasyonu ve Karar Öneri Mekanizması
### 4.1. Embedding ve GraphDB
Her mahkeme kararı, içeriğine göre text ve embedding vektörleri ile Neo4j'de Karar düğümü olarak saklanır.


LLM tarafından sorulan doğal dil sorular embedding aracılığıyla en benzer kararlarla eşleştirilir.


### 4.2. QA Chain ve Agent Tabanlı Seçim
Kullanıcı, arayüz üzerinden iki yöntemden birini seçebilir:


QA Chain: Soruya benzer kararlar getirilir ve doğrudan içerik yanıtlanır.


ReAct Agent: LangChain ReAct ajanı Neo4j’e doğal dil üzerinden Cypher üreterek yanıt oluşturur.

## 5.  Kullanıcı Arayüzü
Çok aşamalı (multi-step) form yaklaşımı:


Risk Tahmini (ML)


Suç Tipi ve Nitelikli Haller


İndirim / İstisna Sebepleri


Hakim Kararı Girişi


Model Analizi ve Öneriler


Ek sekmeler:


Legal References: TCK 25-34, 61, 81-82 açıklamaları


Model Info: Teknik detaylar ve uyarılar


Karar Sorgulama: LLM destekli karar arama



## 6.  Veri Modeli
### 6.1. Düğümler

Örnek bir Karar node’u ve bağlı olduğu diğer node’lar:

Karar: Mahkeme kararı (metin, embedding, madde, nitelik, ceza türü vs.)


Sanık, Suç, Madde, İndirim, Nitelikli Hâl gibi düğümler isteğe bağlı olarak genişletilebilir


### 6.2. İlişkiler
🔹 Sanık ile İlişki:
(s:Suspect)-[:YER_ALIR]→(k:Karar)
 → Sanığın yer aldığı karar


🔹 Suç maddesi (TCK maddesi) ile:
(k:Karar)-[:DAYANIR]→(m:Madde)
 → Kararın dayandığı TCK maddesi (örn. TCK 81, TCK 82)


🔹 Nitelikli Haller (örneğin: planlama, canavarca his):
(k:Karar)-[:NITELIK_TASIR]→(n:NitelikliHal)
 → Kararın içerdiği nitelikli haller


🔹 Hafifletici Sebepler:
(k:Karar)-[:INDIRIM_ICERIR]→(i:Indirim)
 → Kararda uygulanan indirici (hafifletici) sebepler


🔹 Risk Tahmini:
(s:Suspect)-[:TAHMINE_GORE]→(r:Risk)
 → Sanığın tahmini recidivism (yeniden suç işleme) riski


🔹 Hakimin verdiği ceza:
(k:Karar)-[:CEZA_OLARAK]→(c:Ceza)
 → Kararda verilen ceza (örneğin "müebbet", "25 yıl")


🔹 Olay Özeti:
(k:Karar)-[:ICERIR]→(o:Olay)
 → Kararın içerdiği olay özeti metni









