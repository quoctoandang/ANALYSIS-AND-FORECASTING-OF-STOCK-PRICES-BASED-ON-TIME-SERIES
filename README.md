# Stock Price Analysis and Forecasting using SARIMAX and Machine Learning Models

[![Project Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)](https://github.com/your-github-username/your-repo-name)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) <!-- Optional: Add your license file and link -->

## Overview

This project is the graduation thesis conducted at **Industrial University of Ho Chi Minh City, Faculty of Information Technology**.  It focuses on the analysis and forecasting of stock prices, specifically for **TESLA (TSLA)**, using time series analysis and machine learning techniques.

The thesis explores the predictive power of different models, including:

* **SARIMAX (Seasonal Autoregressive Integrated Moving Average with Exogenous Variables)**: A statistical time series model that accounts for seasonality and external factors.
* **LSTM (Long Short-Term Memory)**: A type of Recurrent Neural Network (RNN) suitable for sequential data.
* **Random Forest**: An ensemble learning method based on decision trees.
* **XGBOOST (Extreme Gradient Boosting)**: A gradient boosting algorithm known for its performance and speed.
* **Combined SARIMAX + XGBOOST**: A hybrid model leveraging the strengths of both statistical and machine learning approaches.

The research aims to build a comprehensive forecasting model by integrating various data sources and analytical perspectives to improve the accuracy and reliability of stock price predictions.

## Table of Contents

- [Overview](#overview)
- [Models Used](#models-used)
- [Data Sources](#data-sources)
- [Key Features & Findings](#key-features--findings)
- [Installation](#installation) <!-- Optional, if you have code to run -->
- [Results](#results)
- [Authors](#authors)
- [License](#license) <!-- Optional -->

## Models Used

This project implemented and compared the following models for stock price forecasting:

* **SARIMAX**:  Utilized for its ability to model time series data with seasonality and exogenous variables, capturing linear patterns and seasonal components in stock prices.
* **LSTM**: Employed to capture complex, non-linear dependencies in time series data, potentially learning long-term patterns in stock price movements.
* **Random Forest & XGBOOST**: Explored as machine learning baselines, leveraging their ability to handle diverse data types and capture non-linear relationships.
* **SARIMAX + XGBOOST**:  Developed as a hybrid model to combine the strengths of SARIMAX in handling linear time series components and XGBOOST in capturing non-linear residuals, aiming for improved forecasting accuracy.

## Data Sources

To build a robust predictive model, the thesis utilized a multi-faceted dataset derived from:

* **Yahoo Finance (YF)**:  Historical stock price data for TESLA, including:
    * Open, High, Low, Close prices
    * Adjusted Close Price
    * Volume
* **EODHD API**:  News article data related to TESLA, used for sentiment analysis.
    * Article Titles and Content
    * Publication Dates
    * Source Categories
    * Sentiment Scores (analyzed using both API provided sentiment and a custom model - Meta-LLaMA via DeepInfra)
* **Yahoo Finance (YF) - Financial Statements**: Fundamental financial data for TESLA, including:
    * Total Revenue
    * Gross Profit
    * Operating Income
    * Net Income
* **Technical Indicators**: Calculated from stock price data to capture market dynamics and trends:
    * EMA, RSI, MACD, Bollinger Bands, SMA, etc.
    * Time-based features (year, month, day, day of week, is_weekend, month_end)

## Key Features & Findings

* **Comprehensive Dataset**:  Integrated technical, fundamental, and sentiment analysis data to create a multidimensional dataset for improved stock price prediction.
* **Superior Performance of SARIMAX and SARIMAX + XGBOOST**: Empirical results demonstrated that both SARIMAX and the combined SARIMAX + XGBOOST models outperformed other machine learning models (LSTM, Random Forest, XGBOOST) in predicting TESLA's closing stock price.
* **High Accuracy and Reliability**:  SARIMAX and SARIMAX + XGBOOST achieved impressive performance metrics (RMSE, R², MAE, MAPE), indicating higher accuracy and reliability in stock price forecasting compared to other methods.
* **Practical Implications**: The research provides valuable insights and potentially equips stock investors with intelligent decision-making tools to minimize investment risk and optimize returns in the dynamic stock market environment.

## Installation

**Required Libraries:**

* Python (>= 3.7 recommended)

* pandas

* numpy

* statsmodels

* scikit-learn (sklearn)

* tensorflow/keras (if using LSTM)

* xgboost

* yfinance

* requests (for EODHD API)

... (add any other libraries you used, create a requirements.txt file for easy installation)

For detailed usage instructions and code execution, please refer to the thesis document.

## Results
The detailed results, including performance metrics and visualizations, are available in the thesis document [link to your thesis document if hosted online, or mention its location in the repository].

Key performance metrics for the best models (SARIMAX and SARIMAX + XGBOOST) are summarized in the thesis abstract and documentation.

**Authors:**
* ***Châu Mỹ Uyên*** - 20087481

* ***Đặng Quốc Toàn*** - 20051051

**Supervisors:**:

* Instructor 1: TS. Nguyễn Chí Kiên (PhD. Nguyen Chi Kien)

* Instructor 2: TS. Vũ Đức Thịnh (PhD. Vu Duc Thinh)

* Industrial University of Ho Chi Minh City
* Faculty of Information Technology
* December 2024

## License
<!-- **Choose a license if you want to specify how others can use your work. A common open-source license is MIT License. If you choose to use one, add a LICENSE file to your repository and update the badge at the top.** -->
This project is licensed under the MIT License - see the LICENSE file for details. <!-- Remove this line if you don't include a license -->

Feel free to adapt this README to better reflect your project and add more specific details. Good luck with your thesis project!
