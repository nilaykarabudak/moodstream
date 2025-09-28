import streamlit as st
import joblib
import time
import random
import pandas as pd
import matplotlib.pyplot as plt
from os import path

def modelleri_yukle():
    """Modelleri yüklerken progress bar ile görsel feedback verelim"""

    model_dosyalari = {
        'kategori': 'kategori_model.pkl',
        'altkategori': 'altkategori_model.pkl',
        'segment': 'segment_model.pkl',
        'sentiment': 'sentiment_model.pkl',
        'vektor_kategori': 'kategori_vectorizer.pkl',
        'vektor_altkategori': 'altkategori_vectorizer.pkl',
        'vektor_segment': 'segment_vectorizer.pkl',
        'vektor_sentiment': 'sentiment_vectorizer.pkl'
    }

    models = {}
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, (isim, dosya) in enumerate(model_dosyalari.items()):
        try:
            if not path.exists(dosya):
                raise FileNotFoundError(f"{dosya} bulunamadı!")

            models[isim] = joblib.load(dosya)
            durum = f"✔ {dosya} yüklendi (%{int((i+1)*100/len(model_dosyalari))})"
            status_text.text(durum)
            progress_bar.progress((i+1)/len(model_dosyalari))

            time.sleep(random.uniform(0.3, 0.7))

        except Exception as e:
            st.error(f"⛔ Hata: {str(e)}")
            st.stop()

    status_text.text("Tüm modeller başarıyla yüklendi!")
    time.sleep(0.5)
    return models

st.set_page_config(
    page_title="MoodStream: Yayın Analiz Asistanı",
    page_icon="🎬",
    layout="centered"
)

st.markdown("""
    <h1 style='text-align: center; color: #00e0d5; font-size: 40px;'>
        MoodStream: Yayın Analiz Asistanı
    </h1>
    <p style='text-align: center; color: #cccccc;'>
        Dijital yayın platformlarına gelen yorumları anında analiz edin.
    </p>
""", unsafe_allow_html=True)

yorum = st.text_area(
    "Yorumunuz:", 
    height=120,
    placeholder="Buraya yazın...",
    key="yorum_girisi"
)

if st.button("ANALİZ ET", type="primary"):
    if not yorum.strip():
        st.warning("⚠️ Lütfen yorum giriniz!")
    else:
        with st.spinner("Analiz başlıyor..."):
            time.sleep(1)

            progress_text = st.empty()
            progress_bar = st.progress(0)

            for i in range(5):
                progress_bar.progress((i+1)*20)
                durumlar = [
                    "Metin temizleniyor...",
                    "Vektöre çevriliyor...",
                    "Kategori tahmini...",
                    "Duygu analizi...",
                    "Sonuçlar hazırlanıyor..."
                ]
                progress_text.text(durumlar[i])
                time.sleep(random.uniform(0.2, 0.5))

            try:
                models = modelleri_yukle()

                sonuclar = {
                    'KATEGORİ': models['kategori'].predict(models['vektor_kategori'].transform([yorum]))[0],
                    'ALT KATEGORİ': models['altkategori'].predict(models['vektor_altkategori'].transform([yorum]))[0],
                    'SEGMENT': models['segment'].predict(models['vektor_segment'].transform([yorum]))[0],
                    'DUYGU': models['sentiment'].predict(models['vektor_sentiment'].transform([yorum]))[0]
                }

                st.balloons()
                st.success("ANALİZ TAMAMLANDI!")
                st.markdown("---")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"""
                        **🧩 KATEGORİ**  
                        `{sonuclar['KATEGORİ']}`  

                        **🔍 ALT KATEGORİ**  
                        `{sonuclar['ALT KATEGORİ']}`
                    """)

                with col2:
                    st.markdown(f"""
                        **🏷️ SEGMENT**  
                        `{sonuclar['SEGMENT']}`  

                        **😊 DUYGU**  
                        `{sonuclar['DUYGU']}`
                    """)

                st.markdown("---")
                fig, ax = plt.subplots(figsize=(8, 3))
                pd.Series(sonuclar).value_counts().plot(
                    kind='bar',
                    ax=ax,
                    color=['#FF6B6B', '#4ECDC4', '#45B7D1']
                )
                plt.xticks(rotation=15)
                st.pyplot(fig)

            except Exception as e:
                st.error(f"❌ Analiz sırasında hata: {str(e)}")


