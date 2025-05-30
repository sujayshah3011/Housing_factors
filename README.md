# Analyzing Factors Influencing US Home Prices (2005-2025)

## Project Overview

This project investigates the key economic factors affecting US home prices over the period 2005-2025. It utilizes the Case-Shiller U.S. National Home Price Index (`CSUSHPISA`) as the primary target variable and analyzes its relationship with various economic indicators. The analysis involves:

1.  **Data Aggregation & Preprocessing**: Merging multiple time-series datasets and cleaning the data.
2.  **Exploratory Data Analysis (EDA)**: Visualizing trends and correlations.
3.  **Model Building**: Developing regression models (Linear Regression and Random Forest) to quantify the impact of different factors.
4.  **Model Evaluation & Interpretation**: Assessing model performance and deriving insights from feature importance and coefficients.

The goal is to identify significant drivers of home price fluctuations and understand their quantitative impact.

## Data Sources

The project uses time-series data for several US economic indicators, typically sourced from economic data providers like FRED (Federal Reserve Economic Data). The raw data is provided in individual CSV files located in the repository, including:

*   `CSUSHPISA.csv`: Case-Shiller U.S. National Home Price Index (Target Variable)
*   `MEHOINUSA672N.csv`: Real Median Household Income in the United States
*   `WPUSI012011.csv`: Producer Price Index by Commodity: Inputs to Industries: Net Inputs to Residential Construction, Goods (Construction Costs)
*   `UNRATE.csv`: Unemployment Rate
*   `HOUST.csv`: Housing Starts: Total: New Privately Owned Housing Units Started
*   `CPIAUCSL.csv`: Consumer Price Index for All Urban Consumers: All Items in U.S. City Average (Inflation)
*   `FEDFUNDS.csv`: Federal Funds Effective Rate (Interest Rates)
*   `POPTHM.csv`: Population
*   `A939RX0Q048SBEA.csv`: Real Disposable Personal Income

## Methodology

The analysis is primarily conducted through two Jupyter Notebooks:

### 1. `merged_dataset.ipynb`: Data Merging and Initial Preprocessing
   *   Collects all individual raw CSV data files from the specified folder.
   *   Converts `observation_date` columns to datetime objects.
   *   Merges these datasets into a single comprehensive DataFrame based on `observation_date` using an outer join to preserve all data points.
   *   Sorts the merged data by date.
   *   Handles missing values resulting from the merge and varying frequencies of original datasets using linear interpolation, followed by forward-fill and backward-fill strategies.
   *   Saves the resulting cleaned and merged dataset as `final_data.csv`.

### 2. `hello.ipynb`: Detailed Analysis, EDA, and Modeling
   *   **Data Preprocessing**:
      *   Loads the `final_data.csv`.
      *   Performs an initial check for missing values.
      *   Analyzes correlations between `_x` and `_y` suffixed columns (if any, resulting from different data vintages or transformations in original sources) to select the most relevant features and avoid multicollinearity. For this project, `_x` columns are primarily used.
      *   Selects features for modeling and defines the target variable (`CSUSHPISA_x`).
      *   Scales numerical features using `StandardScaler` to prepare for regression modeling.
   *   **Exploratory Data Analysis (EDA)**:
      *   Plots the trend of the Case-Shiller Home Price Index over time.
      *   Generates a correlation heatmap to visualize relationships between the selected economic features and home prices.
      *   Saves these plots as `home_price_trend.png` and `correlation_heatmap.png`.
   *   **Model Building**:
      *   Splits the scaled data into training (80%) and testing (20%) sets.
      *   Trains two regression models:
          *   **Linear Regression**: To provide interpretable coefficients regarding the direction and magnitude of each factor's impact.
          *   **Random Forest Regressor**: To capture potential non-linear relationships and provide robust feature importance measures.
   *   **Model Evaluation and Interpretation**:
      *   Evaluates both models on the test set using R² (coefficient of determination) and Mean Squared Error (MSE).
      *   Extracts and plots feature importances from the Random Forest model, saving it as `feature_importance.png`.
      *   Examines the coefficients from the Linear Regression model.
      *   Provides an interpretation of the results, discussing model performance and the economic significance of key features.

## Key Files in the Repository

*   `merged_dataset.ipynb`: Notebook for merging and preparing the `final_data.csv`.
*   `hello.ipynb`: Notebook for the main analysis, EDA, modeling, and interpretation.
*   `final_data.csv`: The consolidated, preprocessed dataset used for the analysis in `hello.ipynb`.
*   Individual `*.csv` files (e.g., `CSUSHPISA.csv`, `MEHOINUSA672N.csv`, etc.): Raw input data.
*   `home_price_trend.png`: Output plot showing US home price trends.
*   `correlation_heatmap.png`: Output plot showing the correlation matrix.
*   `feature_importance.png`: Output plot showing feature importances from the Random Forest model.
*   `README.md`: This file.

## How to Use / Reproduce Results

1.  **Prerequisites**:
    *   Python 3.x
    *   Jupyter Notebook or JupyterLab environment.
    *   Required Python libraries:
        *   `pandas`
        *   `numpy`
        *   `matplotlib`
        *   `seaborn`
        *   `scikit-learn`
        *   `glob` (standard library)
        *   `os` (standard library)

    You can install the necessary libraries using pip:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn jupyterlab
    ```

2.  **Setup**:
    *   Clone this repository:
        ```bash
        git clone https://github.com/sujayshah3011/Housing_factors.git
        cd Housing_factors
        ```
    *   Ensure all raw `.csv` data files are present in the root directory (or update `folder_path` in `merged_dataset.ipynb` if they are located elsewhere).

3.  **Running the Notebooks**:
    *   **Step 1: Generate the merged dataset.**
        Open and run all cells in `merged_dataset.ipynb`. This will read the individual CSVs, merge them, and save `final_data.csv`.
    *   **Step 2: Perform the analysis and modeling.**
        Open and run all cells in `hello.ipynb`. This notebook uses the `final_data.csv` generated in the previous step to perform EDA, train models, evaluate them, and generate output plots and insights.

## Expected Insights & Visualizations

The analysis aims to provide quantitative answers to how factors such as:
*   Median Household Income (`MEHOINUSA672N_x`)
*   Construction Costs (`WPUSI012011_x`)
*   Unemployment Rate (`UNRATE_x`)
*   Housing Starts (`HOUST_x`)
*   Inflation (CPI - `CPIAUCSL_x`)
*   Federal Funds Rate (`FEDFUNDS_x`)
*   Population (`POPTHM_x`)
*   Real Disposable Income (`A939RX0Q048SBEA_x`)

...correlate with and predict the Case-Shiller Home Price Index.

Key visualizations generated include:
*   **Home Price Trend (`home_price_trend.png`)**: Shows the historical movement of US home prices.
*   **Correlation Heatmap (`correlation_heatmap.png`)**: Highlights linear relationships between variables.
*   **Feature Importance Plot (`feature_importance.png`)**: Ranks the influence of each economic factor on home price predictions according to the Random Forest model.

## Future Work

Potential areas for future improvement include:
*   Incorporating lagged variables to account for time delays in economic impacts.
*   Utilizing more advanced time-series models (e.g., ARIMA, VAR, or Prophet).
*   Experimenting with other machine learning models like Gradient Boosting or Neural Networks.
*   Exploring interaction effects between different economic features.
*   Performing more granular regional analysis if data permits.
