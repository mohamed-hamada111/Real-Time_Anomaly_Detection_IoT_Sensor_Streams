# Real‑Time Anomaly Detection for IoT Sensor Streams
Anomify_Project/
│
├── data/                   # فولدر الداتا (مش بيترفع على جيت هاب)
│   ├── raw/                # الداتا الأصلية (normal.csv, attack.csv)
│   └── processed/          # الداتا بعد التنظيف والـ Feature Engineering
│
├── configs/                # فولدر الإعدادات
│   └── config.yaml         # هنحط هنا كل المتغيرات (Paths, Window Size, Thresholds)
│
├── src/                    # قلب المشروع (الكود الفعلي)
│   ├── __init__.py
│   ├── data_loader.py      # سكريبت بيقرا الداتا ويدمجها
│   ├── preprocess.py       # سكريبت بينضف الداتا (Missing values, Types)
│   ├── features.py         # سكريبت بيحسب الـ Rolling Stats والـ Scaling
│   ├── model.py            # الكلاس بتاع الـ Autoencoder أو الموديل اللي هنختاره
│   └── evaluate.py         # سكريبت بيحسب الـ Precision/Recall والـ F1-Score
│
├── pipelines/              # تجميع للسكريبتات اللي فوق
│   ├── train_pipeline.py   # بياخد الداتا من الصفر لحد ما يسيف الموديل
│   └── inference.py        # سكريبت بياخد قراءة واحدة من السينسور ويطلع Anomaly ولا لأ
│
├── app/                    # فولدر الـ API (عشان الـ Deployment)
│   └── main.py             # الـ FastAPI اللي هيستقبل الـ Streaming Data
│
├── notebooks/              # هنرمي هنا كل النوت بوكس بتاعت الـ EDA (زي اللي إنت عملتها)
│
├── requirements.txt        # المكتبات المستخدمة
└── Dockerfile              # عشان نـ Containerize المشروع بعدين على Azure
