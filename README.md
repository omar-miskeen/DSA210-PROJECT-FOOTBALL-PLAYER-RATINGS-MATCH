## ⚽ Project: Predicting FC Barcelona's Match Outcomes via Key Attacking Player Metrics (2024-2025)

### Overview

This project is a data-driven investigation into the relationship between the on-pitch performance of four core FC Barcelona attacking players: Robert Lewandowski, Lamine Yamal, Raphinha, and Pedri, and the resulting team outcomes during the 2024-2025 season.

We will move beyond simple goals and assists (G+A) to utilize *advanced metrics* (like Shot-Creating Actions, Progressive Passes, and Possessions Regained) collected on a per-match basis. The analysis has two core goals:

1. To build predictive regression models to understand how these advanced statistics influence the subjective SofaScore User Rating for each player.

2. To build a classification model that determines the extent to which the combined performance of these four players predicts the team's final result (Win, Draw, or Loss) for that game. This component addresses the project requirement for data enrichment and originality by linking individual output to collective success.

### Motivation

As a fan of Barcelona and a fan of their llast season's sucesses, I've observed that subjective performance ratings and match narratives often oversimplify player impact. The quote "these players had the biggest affects on last season's great performance" highlights the intuition that a few key individuals often drive team results.

This project aims to use objective data to prove or disprove this notion. By isolating the contribution of these four key attacking players, the analysis will:

* Quantify which specific actions for example, Pedri's Progressive Carries or Lewandowski's xG truly matter to their individual match rating.

* Determine if the collective attacking output of these four players is a statistically significant predictor of whether FC Barcelona wins or drops points, providing a deeper analytical view than traditional team-level metrics.

### Data Sources & Collection Plan

All data for the 2024-2025 season will be collected on a per-match basis, ensuring the most granular level of analysis.

| Data Type | Source | Collection Method | Purpose | 
 | ----- | ----- | ----- | ----- | 
| **Advanced Player Metrics** | Fbref | Web Scraping using Python (BeautifulSoup) | Provides detailed event data (SCA, GCA, xG, defensive actions, possession). | 
| **User Performance Ratings** | SofaScore | Web Scraping using Python (BeautifulSoup) | Provides the target variable (Rating) for the individual player regression model. | 
| **Match Results & Details** | Fbref / SofaScore | Web Scraping / Merging | Provides the target variable (Win, Draw, Loss) for the team classification model. | 

The data collection will focus on extracting and merging all relevant statistics for *Lamine Yamal, Raphinha, Pedri, and Robert Lewandowski* across all official club competitions. The scraped data will be standardized and merged into a unified dataset for modeling.

### 🛠️ Tools Used

* **Python:** The primary language for all data processing, analysis, and modeling.

* **Pandas & NumPy:** Essential for efficient data cleaning, merging the Fbref and SofaScore data, and numerical feature engineering.

* **Scikit-learn:** Used for implementing regression models (to predict individual ratings) and classification models (to predict match outcomes).

* **Matplotlib / Seaborn:** Used for Exploratory Data Analysis (EDA) and creating effective data visualizations of correlations and model findings.

### ❓ Key Questions Explored

1. **Individual Rating Drivers:** For each player, which metrics (e.g., SCA, PrgC, xAG) are the strongest predictors of their per-match SofaScore rating? Is the rating explained primarily by goal contributions, or by less visible creative and progressive actions?

2. **Collective Predictive Power:** Does a model using only the aggregated advanced metrics of these four players perform better than a random baseline in predicting the match outcome (Win/Draw/Loss)?

3. **Role Comparison:** How do the metrics that drive Pedri's rating (as a midfielder) differ from the metrics that drive Lewandowski's rating (as a striker)?

4. **Metric Distribution:** How are the key advanced metrics (xG, xAG, SCA, PrgC) distributed across the season for each player, and are these distributions correlated with periods of high/low team performance?

### 📈 Metrics Utilized (Key Features)

The analysis will leverage the following metrics, collected on a per-match basis:

| Category | Metrics (Selected) | 
 | ----- | ----- | 
| **Dependent Variables (Targets)** | SofaScore Rating, Match Result (Win/Draw/Loss) | 
| **Playing Time** | Min (Minutes Played) | 
| **Offensive Production** | Gls, Ast, PK, PKatt, Sh (Shots Total), SoT (Shots on Target) | 
| **Expected/Advanced** | xG, npxG, xAG, SCA (Shot-Creating Actions), GCA (Goal-Creating Actions) | 
| **Passing & Creativity** | Cmp, Att, Cmp%, PrgP (Progressive Passes) | 
| **Possession & Ball Control** | Touches, Carries, PrgC (Progressive Carries), Take-Ons Attempted, Successful Take-Ons | 
| **Defensive & Discipline** | Tkl (Tackles), Int (Interceptions), Blocks, CrdY (Yellow Cards), CrdR (Red Cards) | 

### ✅ Expected Results

* **Rating Predictability:** I expect to find that different metrics drive the ratings for each player. For instance, Lewandowski's rating may be heavily influenced by *xG* and *SoT*, while *Pedri's* rating may be more sensitive to *PrgC* and *Prgp*.

* **Outcome Prediction:** The classification model that aggregates the four players' stats is expected to show a strong correlation with the final match outcome, particularly in determining draws versus wins, demonstrating the critical dependency of team success on this core attacking unit.

* **Enrichment:** The combined use of two scraped datasets (Fbref metrics + SofaScore ratings) will satisfy the data enrichment requirements.

### Conclusion

This project is a comprehensive application of regression and classification techniques to a real-world sports problem. By successfully scraping and merging detailed player data, I aim to create a robust model that not only predicts individual player perception but also quantifies the collective impact of key attacking talent on FC Barcelona's success in the 2024-2025 season.
