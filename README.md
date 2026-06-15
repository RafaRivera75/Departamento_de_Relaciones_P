# 💬 Análisis de Sentimiento — Reseñas Amazon Alexa

Modelo de **NLP y clasificación de texto** para predecir automáticamente el sentimiento (positivo/negativo) de reseñas de clientes sobre Amazon Alexa, comparando Naive Bayes vs Regresión Logística.

---

## 🎯 Objetivo

Permitir al departamento de Relaciones Públicas identificar automáticamente si los clientes están satisfechos con el producto, sin revisar manualmente cientos de valoraciones en texto libre.

---

## 📂 Dataset

- **Archivo:** `amazon_alexa.tsv`
- **Fuente:** [Amazon Alexa Reviews — Kaggle](https://www.kaggle.com/datasets/sid321axn/amazon-alexa-reviews)
- **Columnas clave:** `rating`, `date`, `variation`, `verified_reviews`, `feedback`
- **Target:** `feedback` (1 = positivo / 0 = negativo)
- **Observación:** Dataset desbalanceado — predominancia de reseñas positivas, lo que afecta la detección de sentimiento negativo

---

## ⚙️ Pipeline del Proyecto

### 1. EDA y Visualización
- Distribución de `feedback` y `rating`
- Relación entre variación del producto y rating promedio
- **Word Clouds** para reseñas generales, positivas y negativas
- Análisis de longitud de reseñas

### 2. Preprocesamiento NLP
- Eliminación de signos de puntuación y **stopwords** con NLTK
- Codificación one-hot de la columna `variation` con `pd.get_dummies`
- Vectorización de texto con **CountVectorizer** (Bag of Words)
- Drop de columnas irrelevantes: `rating`, `date`, `length`

### 3. Modelos Entrenados

| Modelo | Precisión General | Recall clase negativa (0) | Recall clase positiva (1) |
|---|---|---|---|
| Naive Bayes Multinomial | 93% | **0.43** 🏆 | Alto |
| Regresión Logística | **94%** | 0.35 | Alto |

---

## 📊 Hallazgos Principales

- Ambos modelos logran alta precisión general (93-94%), pero **fallan en detectar sentimiento negativo** por el desbalance del dataset
- **Naive Bayes supera a Regresión Logística** en recall para clase negativa (0.43 vs 0.35) — resultado contraintuitivo pero real
- Las Word Clouds revelaron términos claramente diferenciados entre reseñas positivas y negativas
- El desbalance de clases es la limitación principal; técnicas como **SMOTE o class_weight** mejorarían el recall negativo

---

## 🛠️ Tecnologías

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)
![WordCloud](https://img.shields.io/badge/WordCloud-visualization-blueviolet)
![Pandas](https://img.shields.io/badge/Pandas-data--wrangling-lightgrey?logo=pandas)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)

---

## 🚀 Cómo ejecutar

```bash
# 1. Clonar el repositorio
git clone https://github.com/RafaRivera75/Departamento_de_Relaciones_P.git
cd Departamento_de_Relaciones_P

# 2. Instalar dependencias
pip install pandas numpy matplotlib seaborn scikit-learn nltk wordcloud

# 3. Descargar stopwords de NLTK (se ejecuta dentro del notebook)
# nltk.download('stopwords')

# 4. Colocar el dataset en la raíz del proyecto
# Archivo esperado: amazon_alexa.tsv

# 5. Abrir el notebook
jupyter notebook
```

---

## 📁 Estructura del Proyecto

```
Departamento_de_Relaciones_P/
│
├── Departamento_de_Relaciones_P.ipynb   # Notebook principal
├── amazon_alexa.tsv                      # Dataset de entrada
└── README.md
```

---

## 👤 Autor

**Rafael Rivera** — [github.com/RafaRivera75](https://github.com/RafaRivera75)
