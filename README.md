# Transformer For Time Series Data
This is an experimental project about implementation of transformer architecture for a time series data. Transformer model and its variation used on this project is made from scratch using pytorch. The main goal for this project is to self-discover the potential of transformer for time series data without looking at ongoing researches and solve the occuring problem using methods that I think of.

## GDP Per Capita Prediction (This Branch)
NOTE: This branch is discontinued for undetermined amount of time.


This project is a further exploration and improvement from my previous project (research methodology project, gdp per capita prediction) which can be seen from https://github.com/Nicohim87/GDPPerCapitaPredictionModel. The base idea and preprocessing is mostly taken and modified from that project.

## Project Structure
```
./
├── .gitignore
├── README.md
├── architecture.png
│
├── data
│   ├──global-data-on-sustainable-energy.csv  -> Raw Dataset
│   └── cleaned_dataset.csv                   -> Cleaned Dataset (Used for training, val, and test)
│
├── Preprocessing.ipynb                   -> Preprocessing Code
├── gdp_per_capita_pred.ipynb             -> Model Code (Classic Model)
├── gdp_per_capita_pred_moe.ipynb         -> Model Code (MoE Model)
│
├── transformer_components.py             -> Basic Transformer Components
├── transformer_architecture_base.py      -> Classic Transformer Architecture
└── transformer_architecture_moe.py       -> Transformer Architecture with Mixture of Experts implementation
```

## Dataset
The dataset used is "Global Data on Sustainable Energy (2000-2020)" by Ansh Tanwar. This dataset contains energy and gdp data for around 173 countries accross 21 years.

The dataset can be accessed from this link: https://www.kaggle.com/datasets/anshtanwar/global-data-on-sustainable-energy

## Architecture
![alt text](architecture.png)

## Ideas Tested
- Transformer architecture has two inputs, which are the encoder input and the decoder input.
    - The encoder input is used to input the global average values with time (t0 to t0+n)
    - The decoder input is used to input the local (country specific) values with time (t0 to t0+n)
    - The decoder output is the local prediction with time (t0+1 to t0+n+1)

## Findings
- The dataset used has problems that cannot be fixed
    - The data size is too small
    - The available sequence length for each entity is too short

- Issues on the model after training
    - Model refuses to predict, they just take the data from the previous year as prediction for current year
    - Data which should not change at all (Land area, Latitude, Longitude) changes quite a lot
    - The inability to input constants such as land area, latitude, and longitude directly into the model without being actively predicted by the model