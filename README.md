# 🛡️ LakshmanRekha AI: SVR-Powered Human Tolerance & Attrition Tracker

An end-to-end predictive machine learning framework that models human behavioral limits, tracking cumulative frustration patterns to forecast employee burnout and relationship separation risks before critical boundaries are crossed.

---

## 🧠 Core Philosophy & Algorithm Design
Traditional regression algorithms (like Linear Regression) fail significantly when modeling human corporate behavior. In the real world, human relationships absorb small daily operational frictions up to a point, but trigger immediate structural shifts once a personal threshold is breached.

This tool builds upon Vladimir Vapnik's **Support Vector Regression (SVR)** algorithm to construct a digital **"LakshmanRekha" ($\epsilon$-insensitive loss tube)**:
* **Inside the Safe-Zone Tube ($|y - f(x)| \le \epsilon$):** Baseline daily stress, minor arguments, and operational micro-frustrations are completely absorbed. The algorithm applies **Zero Penalty**, ensuring predictions remain stable and unbothered by normal daily noise.
* **Outside the Safe-Zone Tube ($|y - f(x)| > \epsilon$):** Extreme frustration events (driven by stagnant promotions or toxic management) force data vectors to breach the boundary. The algorithm treats these violators as critical **Support Vectors**, applying strict structural penalties to shift the prediction hyperplane and map high-risk separation metrics accurately.

---

## 🛠️ Data Infrastructure & Variables Architecture
Rather than analyzing static, unidimensional vectors, this pipeline implements multi-variable modeling inspired by **IBM Watson HR Analytics** standards to completely eliminate prediction bias:

* `Daily_Hours` *(Continuous)*: Average daily workload exertion metrics.
* `Experience_Years` *(Continuous)*: Total professional tenure in the ecosystem.
* `Salary_Satisfaction` *(Ordinal: 1 to 5)*: Subjective financial baseline stability.
* `Promotion_Gap_Years` *(Continuous)*: Career progression stagnation index (Major frustration catalyst).
* `Manager_Relation` *(Ordinal: 1 to 5)*: Direct indicator of environmental psychological safety.
* **`Frustration_Score` (🎯 Continuous Target Variable):** The cumulative outcome probability index mapped dynamically from $0\%$ to $100\%$.

---

## 🏗️ System Pipeline Architecture
The project is built around a rigorous 4-tier pipeline ecosystem engineered for high-availability enterprise portfolios:
1. **Simulation Layer (NumPy):** Direct math distribution generation mimicking organic corporate attrition patterns, embedding synthetic structural anomalies (frustrated outliers).
2. **Persistence Layer (SQL/SQLite3):** Relational extraction pipeline filtering critical employee vectors with bad manager matrices sorted directly via query mechanics.
3. **Core ML Layer (Scikit-Learn SVR):** Features preprocessing, normalization via `StandardScaler`, and configuration of an RBF (Radial Basis Function) kernel equipped with a strict error controller ($C=100$) to prioritize extreme boundaries over global averages.
4. **Interface Engine (Streamlit Web UI):** Responsive dark-theme production layout featuring automated calculations, color-coded threshold status indicators, and an instant **Dual-Language Global Selector (English & Hindi/Hinglish)** toggle for accessibility.

---

## 📊 Deployment Artifacts & Verification Matrix
The system is built on an extremely clean, minimal, and fully auditable workspace blueprint:
* `app.py`: Production-ready application front-end.
* `LakshmanRekha_AI_Exploration.ipynb`: Complete exploratory data analysis, visualization, and training notebook.
* `lakshmanrekha_metrics.db`: Pre-seeded structured SQL data.
* `lakshmanrekha_svr_model.pkl`: Serialized RBF Support Vector engine.
* `scaler_x.pkl` & `scaler_y.pkl`: Pre-fitted operational scalar pipelines.
* `requirements.txt`: Locked software package dependency boundaries.

---

## ⚡ Setup & Production Execution
To audit the pipeline metrics locally, execute the following runtime initialization sequences inside your local shell environment:

```bash
# 1. Clone the repository and navigate to the project directory
git clone https://github.com
cd LakshmanRekha-AI

# 2. Deploy environmental boundaries and dependencies
pip install -r requirements.txt

# 3. Initialize the Streamlit interactive framework
streamlit run app.py
```
