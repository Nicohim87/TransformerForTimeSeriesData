# Transformer For Time Series Data
This is an experimental project about implementation of transformer architecture for a time series data. Transformer model and its variation used on this project is made from scratch using pytorch. The main goal for this project is to self-discover the potential of transformer for time series data without looking at ongoing researches and solve the occuring problem using methods that I think of.

## Project Structure
```
./
├── .gitignore
├── README.md
├── architecture.png
│
├── data
│   └──global_air_quality_2014_2025.csv   -> Raw Dataset
│
├── notebooks
│   └──eda.ipynb                          -> Data Exploration
│
├── transformer_components.py             -> Basic Transformer Components
├── transformer_architecture_base.py      -> Classic Transformer Architecture
└── transformer_architecture_moe.py       -> Transformer Architecture with Mixture of Experts implementation
```

## Dataset
The dataset used is "World Air Pollution & AQI Dataset (2014–2025)" by Ashutosh Singh.

The dataset can be accessed from this link: https://www.kaggle.com/datasets/ashyou09/world-air-pollution-and-aqi-dataset-20142025

## Architecture (Not yet updated)
![alt text](architecture.png)

## Ideas Tested
(Not yet started)

## Findings
- Dataset does not have any missing values