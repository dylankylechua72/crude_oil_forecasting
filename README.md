# Crude Oil Price Forecasting Project

## Overview
This project implements machine learning techniques to forecast crude oil prices using historical time series data. The system includes two modeling approaches: XGBoost (a powerful gradient boosting algorithm) and LSTM (a deep learning approach for sequential data).

## Features
- **Multiple Modeling Approaches**: Compare machine learning and deep learning methods
- **Comprehensive Evaluation**: RMSE metrics for model comparison
- **Data Visualization**: Clear plots of actual vs predicted values
- **Scalable Architecture**: Modular design for easy extension

## Technologies Used
- Python 3.8+
- Key Libraries:
  - Pandas (data manipulation)
  - NumPy (numerical operations)
  - Matplotlib (visualization)
  - scikit-learn (machine learning utilities)
  - XGBoost (gradient boosting)
  - TensorFlow/Keras (LSTM implementation)

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/crude-oil-forecasting.git
   cd crude-oil-forecasting
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:
   ```bash
   pip install pandas numpy matplotlib scikit-learn tensorflow xgboost
   ```

## Usage

### Data Preparation
1. Place your crude oil price data in CSV format in the `data/` directory
2. Run the preprocessing script:
   ```bash
   python scripts/data_preprocessing.py
   ```

### Running Models
Run each model independently:

1. **XGBoost Model**:
   ```bash
   python scripts/xgboost_forecast.py
   ```

2. **LSTM Model**:
   ```bash
   python scripts/lstm_forecast.py
   ```

### Expected Output
Each script will:
1. Print the RMSE evaluation metric
2. Generate a plot comparing actual vs predicted values
3. Save visualizations to the `results/` directory

## Sample Results
## Results

### Model Performance
| Model   | Validation Loss (MSE) | RMSE (Scaled Data) | Estimated RMSE ($)* | Training Time |
|---------|-----------------------|--------------------|---------------------|---------------|
| LSTM    | 6.89 × 10⁻⁵           | 0.0083             | 0.42                | ~2 minutes    |

*Assuming crude oil prices range $50-$100. Calculated as:  
`RMSE ($) = RMSE (scaled) × (max_price - min_price) = 0.0083 × 50 = $0.42`

### Key Metrics Interpretation
- **Low Validation Loss**: Final MSE of 6.89 × 10⁻⁵ indicates strong convergence
- **Scaled RMSE**: 0.0083 (on 0-1 normalized data)
- **Dollar RMSE**: ~$0.42 error on $50-$100 price range (~0.84% error)

### How to Verify
1. Check inverse-transformed predictions:
```python
# In your lstm_forecast.py:
predictions = scaler.inverse_transform(predictions)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
rmse_dollars = np.sqrt(mean_squared_error(y_test_inv, predictions))
print(f"Actual RMSE in dollars: ${rmse_dollars:.2f}")

## Customization
To modify the project:

1. **Change model parameters**:
   - Edit the respective script files
   - Key parameters are clearly marked

2. **Add new features**:
   - Economic indicators
   - Weather data
   - Geopolitical events

3. **Extend with new models**:
   - Follow the pattern in existing scripts
   - Add evaluation metrics to the comparison table

## Future Enhancements
- [ ] Add sentiment analysis from news sources
- [ ] Implement ensemble modeling
- [ ] Create API for real-time predictions
- [ ] Develop automated retraining pipeline

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Contact
For questions or suggestions, please contact:
- Dylan Kyle Chua
- Email: dylanchua.2023@scis.smu.edu.sg
- GitHub: [dylankylechua72](https://github.com/dylankylechua72)
