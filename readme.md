Doğrudan https://newmindnaja.streamlit.app/ adresine giderek projeye ulaşabilirsiniz.

1-3300 adet eski hükümlüye ait mock data ile XGB modeli eğitilmiştir.
2-Bu modelin eğitimiyle model tekrara suç işleme riskini hesaplamaktadır. (recidivism)
3-Belli sayıda karar jsonl formatında oluşturulup, karar metinleri vektörize edilerek indexlenip neo4j'e kaydedilmiştir.
4-LLM entegre edilip, LLM'in doğru hareket edebilmesi için TOOL-AGENT'lar oluşturulmuştur.
5-Arayüz hazırlanıp streamlit üzerinden deploy'lanmıştır.