Doğrudan https://newmindnaja.streamlit.app/ adresine giderek projeye ulaşabilirsiniz.

1-3300 adet eski hükümlüye ait mock data ile XGB modeli eğitilip bir pipeline kurulmuştur.

2-Bu modelin eğitimiyle model tekrar suç işleme riskini 0-1-2 (Low-Medium-High) şeklinde hesaplamaktadır. (recidivism)

3-Belli sayıda mahkeme kararı jsonl formatında oluşturulup, karar metinleri vektörize edilerek indexlenip neo4j'e kaydedilmiştir.

4-LLM entegre edilip, LLM'in doğru hareket edebilmesi için TOOL-AGENT'lar oluşturulmuştur.

5-Arayüz hazırlanıp streamlit üzerinden deploy'lanmıştır.