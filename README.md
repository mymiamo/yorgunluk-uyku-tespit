# 🚗 Sürücü Yorgunluk & Dikkat İzleme — Python Prototipi

Kamera tabanlı yüz analizini kullanarak sürücünün **uyku riski (göz kapanma)**, **esneme kaynaklı yorgunluk**, **yana/aşağı uzun bakış (dikkat dağınıklığı)** ve **göz kırpma sıklığı** gibi metrikleri gerçek zamanlı izleyen bir sistem prototipidir.

Bu demo versiyon şunları yapabilir:

- 👁️ **Göz uzun süre kapalı kalırsa (EAR)** → *Uyku tehlikesi alarmı*
- 🥱 **Esneme tespiti (MAR)**
- ↔️ **Baş pozu analizi (Yaw/Pitch deviation)** → uzun süre yana/aşağı bakışta *“Dikkat Dağınık” uyarısı*
- 🔔 **Her olay için ayrı alarm sesi**
- 📊 **Tüm değerlerden birleşik 0–100 arası “Yorgunluk Skoru” (Fatigue Score)** üretme
- 🎥 **Ekranda kamera feed'i ve yüz landmark’ları ile metrikleri gösterme**
- 🖥️ **Monitör tam ekran modunda arayüz dinamik ölçekleme (responsive HUD)**

---

## ✨ Sistem Özellikleri

| Özellik | Açıklama |
|---|---|
| EAR (Eye Aspect Ratio) | Göz kapanmasını ve mikro uyku riskini algılar |
| MAR (Mouth Aspect Ratio) | Esneme davranışını tespit eder |
| Baş Pozi Analizi | Kısa bakışları yok sayar, *3+ saniye sürekli* yan/aşağı bakarsa uyarı verir |
| Yorgunluk Skoru | Göz kapanma + esneme + dikkat dağınıklığı + blink oranı birleşik skor |
| Tam Ekran HUD | Ekran boyuna göre orantılı panel ve yazı ölçeklendirme |
| Alarm Geri Bildirimi | Ses + panel skor barı + yazılı durum değişimi |

---

## 🎯 Çalışma Mantığı

1. Uygulama açılır
2. Kamera tam ekran başlatılır
3. İlk 5 saniye **kişiye özel kalibrasyon** yapılır:
   - Gözler açık
   - Ağız kapalı
   - Yola bakış normal (baş eğik veya sürekli yan bakış yok)
4. Eğer kalibrasyon verisi alınamazsa sistem sabit eşikler ile devam eder.
5. Kısa sağ/sol bakışlar **normal davranış kabul edilir**, uyarı **3 saniye ve sonrası** tetiklenir.

---

## ⚙️ Gereksinimler

### `requirements.txt` içeriği:

