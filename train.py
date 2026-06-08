import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import lunadem

def main():
    print("Loading historical data...")
    # Get the 8 raw features + label
    df = lunadem.get_previously_available_data()
    
    # Feature engineering (basic example)
    print("Extracting features...")
    df['mineral_index'] = lunadem.extract_feature_alpha(df)
    df['thermal_inertia'] = lunadem.extract_feature_beta(df)
    df['albedo_ratio'] = lunadem.extract_feature_gamma(df)
    df['regolith_depth'] = lunadem.extract_feature_delta(df)
    
    # Define features to use
    features = [
        'solar_zenith', 'surface_temp', 'elevation', 'slope', 'reflectance',
        'crater_density', 'sensor_noise_alpha', 'sensor_noise_beta',
        'mineral_index', 'thermal_inertia', 'albedo_ratio', 'regolith_depth'
    ]
    
    X = df[features]
    y = df['label']
    
    print("Training basic Linear Regression model...")
    model = LinearRegression()
    model.fit(X, y)
    
    # Save model to weights_file folder
    os.makedirs('weights_file', exist_ok=True)
    weights_path = os.path.join('weights_file', 'model.pkl')
    joblib.dump(model, weights_path)
    print(f"Model successfully saved to {weights_path}")

if __name__ == "__main__":
    main()
