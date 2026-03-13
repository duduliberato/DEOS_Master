# Long-Term Hourly Energy Consumption Forecasting ⚡

## 📌 Project Overview
This repository contains the core research and implementation for my Master's Thesis. The project focuses on predicting long-term, hourly energy consumption for the Brazilian energy grid using a novel hybrid feature extraction approach.

To validate the model's robustness, a comparative analysis is also performed across diverse energy markets, including **Brazil**, **Sweden**, and **Switzerland**.

---

## 🧪 Methodology: The DEOS-FR Approach
The cornerstone of this research is the **Dynamic Equivalent Oscillation Selector (DEOS)** combined with **Fourier Ramps (FR)**.

### 1. DEOS (Dynamic Equivalent Oscillation Selector)
Used to identify and isolate the most significant cyclical patterns in energy consumption data (daily, weekly, and seasonal cycles). By selecting only the most relevant "equivalent oscillations," we reduce noise and improve the signal-to-noise ratio for the forecasting models.

### 2. FR (Fourier Ramps)
To handle the "non-stationary" nature of energy data (trends and long-term growth), Fourier Ramps are implemented to capture the underlying linear and non-linear trends that standard seasonal decomposition might miss.

---

## 📂 Project Structure
* `Energy_Consumption_Forecast.ipynb`: The main workflow for data preprocessing, model training, and evaluation.
* `DEOS.py`: The core Python module containing the implementation of the DEOS and Fourier Ramps algorithms.
* `BR/`, `Sweden/`, `Switzerland/`: Directories containing country-specific results, plots, and localized configurations.
* `.venv/`: (Local only) Python virtual environment with dependencies (TensorFlow, Pandas, Scikit-learn).

---

## 🚀 Getting Started

### Prerequisites
* Python 3.10+
* Virtual Environment (recommended)

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/duduliberato/DEOS_Master_Research.git](https://github.com/duduliberato/DEOS_Master_Research.git)
