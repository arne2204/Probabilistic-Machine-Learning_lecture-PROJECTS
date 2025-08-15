# Project Report – Football Analytics: Probabilistic Modelling of Expected Goals

## Abstract
This repository contains the results and materials for a Probabilistic Machine Learning project on **expected goals (xG) modelling in football**.  
The project compares Logistic Regression (LR), Bayesian Logistic Regression (BLR), and Bayesian Neural Networks (BNN) for predicting goal probabilities, with emphasis on **probability calibration** and **predictive uncertainty quantification**.

The complete methodology, experiments, and analyses can be found in the in the notebooks folder.

Within the notebooks, all relevant datasets and scripts are linked for reproducibility.

## Project Structure
- **`results/`** – Main project report notebook (`project_report_lukas_pasold.ipynb`) and generated visualisations/tables.  
- **`notebooks/`** – Experimental notebooks and scripts for preprocessing, training, and evaluation.  
- **`data/`** – Processed datasets in `.csv` format (original datasets not included; links provided in the notebook).  

## Data Sources
Data are derived from publicly available football event datasets freely provided by StatsBomb Sources and preprocessing steps are documented in the report and the notebooks.

## Getting Started
**1. Clone the Repository**
```bash
git clone https://github.com/IvaroEkel/Probabilistic-Machine-Learning_lecture-PROJECTS.git
cd Probabilistic-Machine-Learning_lecture-PROJECTS

## Dependencies

You do not need to set up a virtual environment to explore this project.  
The notebooks can be opened interactively in **Google Colab** directly from the GitHub repository, allowing you to run all code without installing anything locally.  
Alternatively, you can clone the repository and open the notebooks in Jupyter Notebook or JupyterLab if you prefer to run them on your own machine.
