## 🔵🔴 Project: Predicting FC Barcelona's Match Outcomes via Key 4 Player Metrics (2024-2025)

### Overview

This project is a data-driven investigation into the relationship between the on-pitch performance of four core FC Barcelona attacking players: Robert Lewandowski, Lamine Yamal, Raphinha, and Pedri, and the resulting team outcomes during the 2024-2025 season.

We will move beyond simple goals and assists (G+A) to utilize *advanced metrics* (like Shot-Creating Actions, Progressive Passes, and Possessions Regained) collected on a per-match basis. 

### Motivation

As a fan of Barcelona and a fan of their last season's sucesses, I've observed that subjective performance ratings and match narratives often oversimplify player impact. The quote "these players had the biggest affects on last season's great performance" highlights the intuition that a few key individuals often drive team results.

This project aims to use objective data to prove or disprove this notion. By isolating the contribution of these four key attacking players, the analysis will:

* Quantify which specific actions for example, Pedri's Progressive Carries or Lewandowski's xG truly matter to their individual match rating.

* Determine if the collective attacking output of these four players is a statistically significant predictor of whether FC Barcelona wins or drops points, providing a deeper analytical view than traditional team-level metrics.

### Data Sources & Collection Plan

All data for the 2024-2025 season will be collected on a per-match basis, ensuring the most granular level of analysis.

| Data Type | Source | Collection Method | Purpose | 
 | ----- | ----- | ----- | ----- | 
| **Advanced Player Metrics** | Fbref | Web Scraping using Python (BeautifulSoup) | Provides detailed event data (SCA, GCA, xG, defensive actions, possession). | 
| **User Performance Ratings** | SofaScore | Web Scraping using Python (BeautifulSoup) | Provides the target variable (Rating) for the individual player analysis. | 
| **Match Results & Details** | Fbref / SofaScore | Web Scraping / Merging | Provides the match outcome (Win, Draw, Loss). | 

The data collection will focus on extracting and merging all relevant statistics for *Lamine Yamal, Raphinha, Pedri, and Robert Lewandowski* across all official club competitions. The scraped data will be standardized and merged into a unified dataset for analysis.

### Tools Used

* **Python:** The primary language for all data processing and analysis.

* **Pandas & NumPy:** Essential for efficient data cleaning, merging the Fbref and SofaScore data, and numerical feature engineering.

* **Matplotlib / Seaborn:** Used for Exploratory Data Analysis (EDA) and creating effective data visualizations of correlations and findings.

* **BeatifulSoup:** Used to data scrape off of these websites.

### Key Questions Explored

1. **Individual Rating Drivers:** For each player, which metrics (e.g., SCA, PrgC, xAG) are the strongest indicators of their per-match SofaScore rating? Is the rating explained primarily by goal contributions, or by less visible creative and progressive actions?

2. **Role Comparison:** How do the metrics that drive Pedri's rating (as a midfielder) differ from the metrics that drive Lewandowski's rating (as a striker)?

3. **Metric Distribution:** How are the key advanced metrics (xG, xAG, SCA, PrgC) distributed across the season for each player, and are these distributions correlated with periods of high/low team performance?

### Metrics Utilized (Key Features)

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

### Expected Results

I expect to find that different metrics drive the ratings for each player. For instance, Lewandowski's rating may be heavily influenced by *xG* and *SoT*, while *Pedri's* rating may be more sensitive to *PrgC* and *Prgp*.

The combined use of two scraped datasets (Fbref metrics + SofaScore ratings) will satisfy the data enrichment requirements.

### Conclusion

This project is a comprehensive application of data collection and exploratory data analysis to a real-world sports problem. By successfully scraping and merging detailed player data, I aim to create a robust dataset that not only visualizes individual player perception but also quantifies the collective impact of key attacking talent on FC Barcelona's success in the 2024-2025 season.
