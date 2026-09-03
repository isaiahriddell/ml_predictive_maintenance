# IoT-Integrated Industrial Predictive Maintenance Pipeline

A robust machine learning project built to predict equipment failures on an industrial production line using high-frequency IoT sensor telemetry.

## Project Overview
Unplanned equipment downtime costs industrial manufacturers billions annually. This project builds an automated predictive maintenance classification pipeline using **Python**, **Pandas**, and **Scikit-Learn** to detect early-stage mechanical faults before catastrophic failure occurs, minimizing unexpected disruptions.

## Key Technical Steps
- **Exploratory Data Analysis (EDA):** Analyzed multi-variable statistical distributions and generated correlation matrices to uncover physical sensor dependencies.
- **Feature Engineering:** Parsed raw time-series strings into structured temporal features (`hour`, `minute`).
- **Categorical Encoding:** Applied **One-Hot Encoding** with `drop_first=True` to eliminate redundant features and avoid the dummy variable trap.
- **Ensemble Modeling:** Trained a **Random Forest Classifier** ($n\_estimators=100$) achieving high predictive accuracy on unseen test data.
- **Model Interpretation:** Extracted **Feature Importances** to validate that predictions align with physical equipment dynamics (prioritizing temperature, electrical current, and vibration).

## Why Random Forest?
- **Robustness to Correlation:** Industrial sensors (such as vibration, temperature, and current) naturally correlate with one another[cite: 3]. Unlike linear models that destabilize under multicollinearity, Random Forests are completely immune to feature correlation.
- **Zero Scaling Requirements:** The algorithm relies entirely on relative threshold splits rather than strict mathematical coefficients, meaning raw, unscaled sensor telemetry can be processed directly without data leakage risks.
- **Built-in Noise Reduction:** By aggregating 100 independent decision trees trained on randomized data subsets, the ensemble naturally dampens single-point sensor anomalies and outlier glitches.

## Results & Insights
- **Model Accuracy:** Achieved high predictive accuracy, demonstrating clear operational boundaries between healthy and degraded states.
- **Physical Validation:** Feature importance scoring revealed that **Temperature** (~34%), **Current** (~27%), and **Vibration** (~22%) were the primary indicators of failure—matching real-world electromechanical failure cascades (mechanical friction increases vibration, which increases current draw to maintain speed, generating excess heat).

## Tech Stack
- **Language:** Python 3.13
- **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
