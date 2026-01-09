## 🔵🔴 Project: Predicting FC Barcelona's Match Outcomes via Key 4 Player Metrics (2024-2025)

### Overview

This project is a data-driven investigation into the relationship between the on-pitch performance of four core FC Barcelona attacking players: Robert Lewandowski, Lamine Yamal, Raphinha, and Pedri, and the resulting team outcomes during the 2024-2025 season.

We move beyond simple goals and assists (G+A) to utilize *advanced metrics* (like Shot-Creating Actions, Progressive Passes, and Progressive Carries) collected on a per-match basis. 

This project investigates the "Chain of Impact" connecting individual performance to team success for FC Barcelona's core attacking unit.

### Motivation

As a fan of Barcelona and their recent success, I've observed that subjective performance ratings and match narratives often oversimplify player impact. The quote "these players had the biggest affects on last season's great performance" highlights the intuition that a few key individuals often drive team results.

This project aims to use objective data to prove or disprove this notion. By isolating the contribution of these four key attacking players, the analysis will:

* Quantify which specific actions for example, Pedri's Progressive Passes or Lewandowski's xG truly correlate with their individual match rating.

* Determine if the collective attacking output of these four players is a statistically significant predictor of whether FC Barcelona wins or drops points, providing a deeper analytical view than traditional team-level metrics.

### Data Sources & Collection Plan

All data for the 2024-2025 season was collected on a per-match basis, ensuring the most granular level of analysis.

| Data Type | Source | Collection Method | Purpose | 
| ----- | ----- | ----- | ----- | 
| **Advanced Player Metrics** | FBref | Web Scraping (Selenium) with Barcelona-only filtering | Provides detailed event data (SCA, xG, defensive actions, possession). | 
| **User Performance Ratings** | SofaScore | Web Scraping (Selenium) via API endpoints | Provides the target variable (Rating) for individual player analysis. | 
| **Match Results & Details** | FBref / SofaScore | Web Scraping / Date-based Merging | Provides the match outcome (Win, Draw, Loss). | 

The data collection focused on extracting all relevant statistics for *Lamine Yamal, Raphinha, Pedri, and Robert Lewandowski* across all official club competitions. The scraped datasets were merged using Player + Date as the key, resulting in **223 match records** spanning **August 17, 2024 to May 25, 2025**.

### Tools & Stack

**Sources:**
- FBref: Advanced event data (SCA, xG, Progressive Actions)
- SofaScore: Match ratings (Target variable)

**Scope:** All official competitions for the 2024-2025 season

**Stack:** 
- Python (Pandas, NumPy) for data processing
- Selenium for web scraping
- Matplotlib/Seaborn for visualization
- SciPy for statistical testing (ANOVA, Shapiro-Wilk, Levene, Pearson/Spearman correlations)
- Scikit-learn, XGBoost, imbalanced-learn for machine learning

### Key Questions Explored

1. **Individual Rating Drivers:** For each player, which metrics (e.g., SCA, progressive_carries, xg_assist) are the strongest indicators of their per-match SofaScore rating? Is the rating explained primarily by goal contributions, or by less visible creative and progressive actions?

2. **Role Comparison:** How do the metrics that drive Pedri's rating (as a midfielder) differ from the metrics that drive Lewandowski's rating (as a striker)?

3. **Temporal Patterns:** Do player ratings show trends over the season? Are there differences in form between the first and second half?

4. **Team Impact:** Do the aggregated attacking metrics of these four players significantly predict match outcomes (Win/Draw/Loss)?

5. **Predictive Modeling:** Can machine learning models accurately predict player ratings and match outcomes using the collected metrics?

### Metrics Utilized (Key Features)

The analysis leveraged the following metrics, collected on a per-match basis:

| Category | Metrics (Selected) | 
| ----- | ----- | 
| **Dependent Variables (Targets)** | Rating, result (W/D/L) | 
| **Playing Time** | Min (Minutes Played) | 
| **Offensive Production** | Gls, Ast, pens_made, pens_att, shots, shots_on_target | 
| **Expected/Advanced** | xg, npxg, xg_assist, sca (Shot-Creating Actions) | 
| **Passing & Creativity** | passes_completed, passes, passes_pct, progressive_passes | 
| **Possession & Ball Control** | touches, carries, progressive_carries, take_ons, take_ons_won | 
| **Defensive & Discipline** | tackles, interceptions, blocks, cards_yellow, cards_red | 

### Methodology

The analysis employed three distinct approaches in sequence:

---

#### **1. Exploratory Data Analysis (EDA): Understanding the Data**

A comprehensive initial exploration to understand distributions, relationships, and patterns in the data.

**Descriptive Analysis:**
- **Data Quality Assessment:** Checked shape (223 rows, 39 columns), missing values (27 missing for advanced metrics), and basic statistics
- **Player Summaries:** Calculated matches played, starts, average ratings, total G/A, and average minutes for each player
- **Result Analysis:** Examined performance patterns across Wins (73%), Draws (12%), and Losses (15%)
- **Correlation Screening:** Computed Pearson and Spearman correlations for all metrics vs Rating
- **Impact Profiling:** Compared average correlations of attacking metrics vs creative metrics per player to identify role-specific tendencies
- **Outlier Detection:** Used IQR method to identify exceptional performances (e.g., perfect 10.0 ratings, hat-tricks)

**Visualizations Created (9 total):**

1. **Rating Distributions (KDE)** - Histogram with kernel density estimation for each player's rating distribution
2. **Correlation Heatmap** - Comprehensive correlation matrix of all key metrics
3. **Violin Plots by Result** - Rating distributions segmented by Win/Draw/Loss outcomes
4. **Attacking Metrics Scatter Plots** - Individual scatter plots per player showing Goals, xG, shots vs Rating
5. **Creative Metrics Scatter Plots** - Individual scatter plots per player showing SCA, progressive passes/carries vs Rating
6. **Discipline Analysis (4-panel)** - Yellow card impact on ratings with statistical testing
7. **Player Comparison Bar Charts** - Side-by-side comparison of Average Rating, Total Goals, Total Assists, Average SCA
8. **Attacking vs Creative Impact Profile** - Bar chart comparing mean absolute correlations for attacking vs creative metrics per player
9. **Rating Trends Over Season** - Time series with 7-game rolling averages showing temporal patterns

---

#### **2. Hypothesis Testing: Theory-Driven Statistical Analysis**

A focused statistical approach testing specific hypothesized relationships.

**Assumption Checks:**
- **Shapiro-Wilk test** for normality of Rating distribution
- **Levene's test** for homogeneity of variance across result types (W/D/L)

**Phase 1 (Metrics → Rating):** 
- Tested **7 pre-selected metrics** per player using Pearson correlations
- **Metrics tested:** npxg, shots, xg_assist, sca, progressive_passes, take_ons_won, progressive_carries
- **Goal:** Identify which specific actions correlate significantly (p < 0.05) with high match ratings for each player
- **Reason:** These 7 metrics represent a balanced mix of finishing quality (npxg, shots), creativity (xg_assist, sca), and ball progression (progressive passes/carries, take_ons won)

**Phase 2 (Team Metrics → Outcome):** 
- **One-way ANOVA** testing if aggregated team metrics differ significantly across Wins, Draws, and Losses
- **Metrics tested:** Total_npxG, Total_SCA, Total_Goals, Avg_Rating, Total_PrgCarries
- **Goal:** Determine if collective attacking output predicts match outcomes

---

#### **3. Machine Learning: Data-Driven Prediction**

A comprehensive modeling approach using all available features to maximize predictive power.

**Feature Selection Strategy:**

The key distinction from hypothesis testing: ML uses **ALL available numeric features** rather than pre-selected metrics.

- **Regression (Rating Prediction):** 
  - Used 27 features (all numeric columns)
  - **Excluded only:** Date, Player, Team, Opponent, result, result_type, Rating (target), Started, Position, venue, Competition, Venue
  - **Reason:** Let algorithms discover which combinations of features best predict ratings
  
- **Classification (Outcome Prediction):** 
  - Used 24 features  
  - **Additional exclusions:** Gls, Ast, Goal_Contribution
  - **Reason:** Prevent data leakage (goals directly determine outcomes)

**Models & Techniques:**

*Regression Models:*
- Ridge Regression (L2 regularization)
- Lasso Regression (L1 regularization, feature selection)
- Random Forest Regressor (ensemble, non-linear)
- XGBoost Regressor (gradient boosting)
- All models tuned via GridSearchCV with 5-fold cross-validation

*Classification Models:*
- Random Forest Classifier (with class balancing)
- XGBoost Classifier (gradient boosting)
- **Class Imbalance Handling:** Tested 3 SMOTE variants (SMOTE, BorderlineSMOTE, SVMSMOTE) to address 73% Win bias
- Tuned via GridSearchCV with 5-fold cross-validation

**Validation Strategy:**

Two complementary approaches to assess model robustness:

1. **Temporal Split (80/20):** Train on first 80% chronologically, test on last 20%
   - Tests real-world prediction (can we predict future performance?)
   - More realistic but harder (data drift, form changes)

2. **Random Split (80/20):** Shuffle data randomly before splitting
   - Tests maximum predictive potential
   - Easier due to similar distributions in train/test

---

### Results

#### **Exploratory Data Analysis Findings**

**Dataset Overview:**
- **223 total match records** from 4 players across all competitions
- **Date range:** August 17, 2024 - May 25, 2025
- **Missing data:** 27 matches missing advanced metrics (12% of dataset)
- **Result distribution:** 163 Wins (73%), 26 Draws (12%), 34 Losses (15%)

**Player Statistics:**

| Player | Matches | Starts | Avg Rating | Goals | Assists | Avg Minutes |
|--------|---------|--------|------------|-------|---------|-------------|
| Lewandowski | 52 | 47 | 7.31 ± 0.83 | 42 | 3 | 75.1 |
| Pedri | 59 | 55 | 7.55 ± 0.58 | 6 | 8 | 78.3 |
| Raphinha | 57 | 34 | 7.91 ± 0.89 | 34 | 22 | 81.6 |
| Yamal | 55 | 51 | 7.79 ± 0.67 | 18 | 21 | 82.4 |

**Performance by Result:**
- **Wins:** Average rating 7.75 (std: 0.76)
- **Draws:** Average rating 7.50 (std: 0.73)
- **Losses:** Average rating 7.24 (std: 0.78)
- Clear trend: Higher ratings correlate with better results

**Top Correlations with Rating (Overall):**
1. xg_assist: 0.544 (Pearson) | 0.588 (Spearman)
2. Gls: 0.543 | 0.467
3. sca: 0.517 | 0.573
4. shots_on_target: 0.471 | 0.410
5. progressive_carries: 0.381 | 0.396

**Impact Profiles (Attacking vs Creative Dominance):**
- **Lewandowski:** Attacking Dominant (finishing metrics drive rating)
- **Raphinha:** Attacking Dominant (goals + creation)
- **Pedri:** Creative Dominant (progression + playmaking)
- **Yamal:** Balanced (equal attacking + creative influence)

**Temporal Trends (From Time Series Visualizations):**
- **Lewandowski & Raphinha:** Ratings **declined** in second half of season (potential fatigue, tactical adjustments)
- **Yamal & Pedri:** Ratings **improved** in second half of season (growing influence, adaptation)
- Suggests role evolution and form fluctuations over the campaign

**Notable Outliers:**
- **Perfect Ratings (10.0):** Raphinha vs Valladolid, Raphinha vs Celta Vigo
- **Hat-tricks (3 goals):** Lewandowski vs Alavés, Raphinha vs Valladolid, Raphinha vs Bayern Munich
- **Exceptional Creativity:** Pedri 16 SCA vs Mallorca, Yamal 13 SCA vs Rayo Vallecano

---

#### **Statistical Hypothesis Testing Results**

**Assumption Testing:**
- **Rating Normality:** W=0.9759, p=0.0007 (Rejected - Non-normal distribution)
- **Variance Homogeneity (W/D/L):** W=0.0389, p=0.9618 (Equal variances across groups)

**Phase 1 - Individual Rating Drivers:**

Different metrics drive each player's rating, confirming **role-specific impacts**:

**Lewandowski (Striker):**
- npxg: r=0.543* (strongest - finishing quality)
- shots: r=0.422*
- xg_assist: r=0.329*
- progressive_carries: r=0.309*
- *Profile: Ratings heavily driven by shooting and finishing*

**Pedri (Midfielder):**
- progressive_passes: r=0.760* (strongest - ball progression)
- xg_assist: r=0.718*
- sca: r=0.651*
- take_ons_won: r=0.518*
- npxg: r=0.341*
- shots: r=0.328*
- *Profile: Creativity and progression dominate, not goal-scoring*

**Raphinha (Winger):**
- sca: r=0.575* (strongest - chance creation)
- xg_assist: r=0.515*
- npxg: r=0.418*
- progressive_passes: r=0.414*
- shots: r=0.354*
- *Profile: Balanced attacking - creates and finishes*

**Yamal (Winger):**
- progressive_carries: r=0.636* (strongest - dribbling)
- xg_assist: r=0.486*
- sca: r=0.410*
- shots: r=0.363*
- *Profile: Ball-carrying and chance creation over finishing*

*Asterisk (*) indicates p < 0.05 (statistically significant)*

**Key Insight:** Ratings are **NOT purely goal-driven**. Creative metrics (SCA, progressive passes/carries, xAG) are equally or more important, especially for midfielders and wingers. Each player is rated based on role-appropriate contributions.

**Phase 2 - Team Impact on Match Outcome:**

ANOVA results showed **limited predictive power** of attacking process metrics:

| Metric | F-statistic | p-value | Significant? |
|--------|-------------|---------|--------------|
| Total_Goals | 6.568 | 0.0027 | ✓ YES |
| Avg_Rating | 6.407 | 0.0031 | ✓ YES |
| Total_npxG | 1.070 | 0.3497 | ✗ No |
| Total_SCA | 0.730 | 0.4863 | ✗ No |
| Total_PrgCarries | 1.239 | 0.2975 | ✗ No |

**Key Insight:** While individual process metrics correlate with ratings, only **actual goals scored** and **average player ratings** significantly differ between wins/draws/losses. Advanced metrics alone don't strongly predict outcomes.

---

#### **Machine Learning Prediction Results**

**Task 1: Rating Prediction (Regression)**

How well can we predict a player's match rating from their performance metrics?

| Split Type | Best Model | R² Score | Performance |
|------------|------------|----------|-------------|
| **Temporal** | Lasso | 0.6518 | Moderate - can predict future ratings with 65% variance explained |
| **Random** | Lasso | 0.7564 | Strong - 76% variance explained when data shuffled |

**Interpretation:** 
- Lasso performs best (automatic feature selection removes noise)
- Temporal split harder (R²=0.65) due to form changes and seasonal trends
- Random split shows maximum potential (R²=0.76)
- Rating prediction is **moderately successful** - metrics explain most but not all rating variance

---

**Task 2: Outcome Prediction (Classification)**

Can we predict Win/Draw/Loss from player metrics alone?

| Split Type | Best SMOTE | Best Model | Accuracy | vs Baseline |
|------------|------------|------------|----------|-------------|
| **Temporal** | BorderlineSMOTE | Random Forest | 68.89% | Baseline: 75.56% (predict all W) |
| **Random** | SVMSMOTE | Random Forest | 75.56% | Baseline: 82.22% (predict all W) |

**Classification Reports:**

*Temporal Split:*
```
              precision    recall  f1-score   support
           D       0.00      0.00      0.00         6
           L       0.29      0.80      0.42         5
           W       0.87      0.79      0.83        34
    accuracy                           0.69        45
```

*Random Split:*
```
              precision    recall  f1-score   support
           D       0.00      0.00      0.00         4
           L       0.25      0.25      0.25         4
           W       0.85      0.89      0.87        37
    accuracy                           0.76        45
```

**Interpretation:**
- Models **underperform baseline** (just predicting "Win" every time is often better)
- Class imbalance is severe (73% Wins) despite SMOTE techniques
- Models struggle with Draws (0% precision/recall) - too rare and contextual
- Loss prediction poor (low precision) - many false positives
- Win prediction decent (85-87% precision) but not better than baseline

**Key Insight:** **Match outcomes are extremely difficult to predict from individual player metrics alone**. This suggests:
1. Team defense, opponent quality, and tactics matter immensely
2. Individual brilliance doesn't guarantee team success
3. Football has inherent randomness and context-dependency
4. Class imbalance (mostly wins) makes meaningful classification nearly impossible

---

### Key Findings Summary

#### What We Learned:

✅ **Individual Performance Metrics Predict Individual Ratings Well**
- Different players rated on different metrics (role-specific)
- Creative actions (SCA, progressive passes/carries) matter as much as goals
- R²=0.65-0.76 for rating prediction shows metrics explain most variance

✅ **Each Player Has a Distinct Impact Profile**
- Lewandowski: Finishing-driven (npxg, shots)
- Pedri: Progression-driven (progressive passes, xAG)
- Raphinha: Balanced attacker (SCA, xAG, npxg)
- Yamal: Dribble-driven (progressive carries, xAG)

✅ **Temporal Patterns Exist**
- Lewandowski/Raphinha declined in second half
- Yamal/Pedri improved in second half
- Suggests adaptation, fatigue, or tactical evolution

❌ **Team Outcomes Cannot Be Predicted from Individual Metrics**
- Only goals scored (not xG) and ratings differ across W/D/L
- Process metrics (xG, SCA, carries) don't predict results
- ML models underperform baseline (just predicting "Win")
- Class imbalance and team complexity limit predictive power

#### The "Chain of Impact" Reality:

**Strong Link:** Metrics → Individual Rating (correlation r=0.3-0.76, p<0.05)

**Weak Link:** Individual Metrics → Team Outcome (ANOVA p>0.05 for process metrics)

**Conclusion:** Individual excellence is **necessary but not sufficient** for team success.

---

### Limitations & Considerations

1. **Sample Size:** 223 match records provides moderate statistical power but limits ML generalization
2. **Class Imbalance:** 73% Wins, 15% Losses, 12% Draws makes outcome prediction extremely difficult
3. **Missing Variables:** Team defense, opponent quality, injuries, tactics, referee decisions not captured
4. **Temporal Dependency:** Football has momentum/form effects, squad rotation, and opponent adaptation not fully modeled
5. **Correlation ≠ Causation:** Significant correlations don't prove metrics cause ratings or outcomes
6. **Rating Subjectivity:** SofaScore ratings, while data-informed, still involve human judgment
7. **Only 4 Players:** Team has 11 players; missing midfield/defensive contributions
8. **Competition Differences:** La Liga, Champions League, Copa del Rey mixed together

---

### Conclusion

This project successfully demonstrates that:

**1. Individual ratings are driven by role-specific metrics**

Strikers are rated on finishing (npxg, shots), midfielders on creativity and progression (progressive passes, xAG, SCA), wingers on a balanced mix (carries, creation, finishing). The SofaScore algorithm clearly values more than just goals and assists where it recognizes the multi-dimensional nature of football performance.

**2. The "eye test" can be quantified**

Advanced metrics like SCA (Shot-Creating Actions) and progressive carries which are often invisible in highlight reels strongly correlate with expert ratings (r=0.41-0.76). These metrics validate the intuition that "good performances" involve more than goal contributions.

**3. Collective process ≠ guaranteed outcomes**

While individual metrics predict individual ratings well (R²=0.65-0.76), aggregated attacking metrics don't significantly predict match results (p>0.05). Only actual goals scored and not expected goals or creativity differ across wins/draws/losses. This highlights that:
- Execution matters more than underlying process
- Defense, opponent quality, and tactics are critical
- Football has inherent unpredictability

**4. Prediction is hard**

ML models struggle to beat baseline predictions (just guessing "Win" every time) for match outcomes. This isn't a model failure, rather it's evidence of football's complexity. Individual brilliance doesn't guarantee team success.

**5. Temporal evolution matters**

Lewandowski and Raphinha's declining ratings vs Yamal and Pedri's improvements in the second half suggest form fluctuations, tactical adjustments, or physical fatigue. Football performance is dynamic, not static.

### Final Thought

The analysis provides **objective evidence** that Lewandowski, Yamal, Raphinha, and Pedri's contributions extend far beyond goals/assists. Each player excels in different dimensions that align with their tactical roles, and these contributions are accurately recognized in performance ratings.

However, **winning requires more than individual excellence**. The weak relationship between individual metrics and team outcomes reminds us that football is the ultimate team sport; 11 players, tactics, momentum, luck, and countless unmeasured factors all contribute to the final result.

The "Chain of Impact" exists, but it's more complex than Metrics → Rating → Outcome. Individual brilliance creates opportunities, but team cohesion and execution determine results.

---

**Dataset:** 223 matches | **Date Range:** Aug 17, 2024 - May 25, 2025 | **Players:** 4 | **Metrics:** 30+ | **Visualizations:** 9
