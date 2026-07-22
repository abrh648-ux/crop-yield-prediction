# 🌾 EthioYield AI — Ethiopian Crop Yield Prediction

A Streamlit web app that predicts crop yield (kg/ha) for Ethiopian regions
using a trained LightGBM machine learning model.

## 📁 Required Files in This Folder

```
streamlit_app/
├── app.py                          ← main app file
├── requirements.txt                ← Python dependencies
├── best_model.pkl                  ← trained LightGBM model  (from Colab Step 8)
├── feature_cols.json               ← feature list            (from Colab Step 8)
├── encodings.json                  ← region/crop encodings   (from Colab Step 8)
├── metrics.json                    ← model performance       (from Colab Step 8)
├── stats.json                      ← dataset statistics      (from Colab Step 8)
└── Etho-Agri Dataset_Enhanced.xlsx ← original dataset (optional, for EDA charts)
```

## 🚀 Deploy to Streamlit Cloud (Free)

1. Push this folder to a GitHub repository
2. Go to https://share.streamlit.io
3. Click **New app**
4. Select your GitHub repo and set **Main file path** to `app.py`
5. Click **Deploy** — live URL in ~2 minutes

## 💻 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
