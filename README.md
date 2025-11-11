
# 💪 Smart Fitness Coach AI

**Smart Fitness Coach** is a Streamlit app that combines machine learning and big-data-style analytics to provide personalized fitness recommendations, a Muscle Growth Index (MGI) predictor, workout planning, and progress tracking.

---

## 🔍 Project Overview

This repository contains a full-featured Streamlit dashboard (`app4.0.py`) and a sample dataset (`fitlife_dataset.csv`) used to train and demo the models. The app includes:
- Personal profile and health dashboard
- MGI (Muscle Growth Index) prediction (RandomForestRegressor)
- Personalized workout planner and daily schedules
- Progress tracker with charts and logs
- ML insights (feature importance, clustering)
- Demonstrations of distributed-data concepts & MapReduce-like pipelines

---

## 📁 Files in this repo

- `app4.0.py` — Main Streamlit application (entry point).
- `fitlife_dataset.csv` — Sample dataset used by the app (place in project root).
- `README.md` — (This file) Project description and instructions.
- `.gitignore` — Suggested entries (see below).

---

## ⚙️ Requirements

Tested with Python 3.9+ (works with 3.8+). Key Python packages:

- streamlit
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Create a `requirements.txt` with:

```
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install with:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to run locally

1. Put `fitlife_dataset.csv` in the same folder as `app4.0.py`.
2. (Optional) Create a virtual environment and install dependencies.
3. Run Streamlit:

```bash
streamlit run app4.0.py
```

Open the local URL shown in your terminal (usually `http://localhost:8501`).

> Note: On first launch the app trains an MGI model from the dataset. Training uses `@st.cache_resource` to speed up subsequent runs, but the first run may take some time depending on your machine.

---

## 🧠 How it works (high level)

- **Data loading**: `fitlife_dataset.csv` is read and preprocessed.
- **ML model**: A RandomForestRegressor is trained to predict a derived `Muscle_Growth_Index`.
- **Personalization**: The user creates a profile that is used to predict MGI and generate workout plans.
- **BDA demos**: Classes like `DistributedDataFrame` and `AdvancedMapReduce` are included to demonstrate large-data processing concepts in a simplified form.
- **Visualization**: Matplotlib + Seaborn for charts; Streamlit to build interactive UI.

---

## 🛠 Tips for GitHub

Suggested `.gitignore` contents:

```
__pycache__/
*.pyc
.env
.venv/
*.egg-info/
.DS_Store
*.csv                # remove this line if you want to keep fitlife_dataset.csv in repo
```

If your dataset is large, consider adding `fitlife_dataset.csv` to `.gitignore` and use Git LFS, or provide a small sample dataset in `data/sample_fitlife.csv`.

---

## 📝 Contributing

Contributions welcome! Steps to contribute:

```bash
git clone <repo-url>
git checkout -b feature/your-feature
# make changes
git add .
git commit -m "Add feature"
git push origin feature/your-feature
# create a PR
```

---

## 📄 License

This project is released under the **MIT License** — feel free to reuse and modify.

---

## 📬 Contact

If you want help improving the README or project structure, reply here and I’ll generate:
- `requirements.txt`
- `.gitignore`
- A GitHub Actions workflow for Streamlit deployment
- A shorter README or labelling for notebooks

Happy coding! 🏋️‍♀️
