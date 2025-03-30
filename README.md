# Anime Platform Recommendation System

## Overview

This project builds a recommender system for an anime streaming platform using both **collaborative filtering** and **content-based methods**. Models are evaluated on traditional accuracy metrics (RMSE, MAE, NDCG) and beyond-accuracy metrics (coverage, personalization, diversity).

## Models

- **Random** and **Popular** recommenders (baselines)
- **Collaborative Filtering**
  - Memory-based (KNNBaseline)
  - Model-based (SVD)
  - AutoSurprise (SVD++)
- **Content-Based**
  - Multi-item and Single-item variants using metadata similarity
- **Ensemble Recommender** combining multiple CF models via Linear Regression

## Evaluation

- Accuracy: RMSE, MAE, NDCG
- Beyond-Accuracy: Coverage, Personalization, Diversity

## Files

- `Anime_Recommendations_Final.ipynb`: Main notebook with models and evaluation
- `Rec_Eng_Data_Exploration.ipynb`: Exploratory Data Analysis
- `data/anime.csv`: Anime metadata
- `data/anime_ratings.csv`: User ratings
- `anime_ratings.csv`: Subset used for the analysis.


## Authors

Jorge Herrero, Zhen Yi Hu Chen, Laura Bradford, Valeria Ulloa, Etienne Descombes
