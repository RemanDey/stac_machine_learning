import pandas as pd
import joblib
import lunadem

def main():
    print("Loading current live telemetry data...")
    # get_current_data doesn't have labels
    df = lunadem.get_current_data()
    
    # Feature extraction
    df['mineral_index'] = lunadem.extract_feature_alpha(df)
    df['thermal_inertia'] = lunadem.extract_feature_beta(df)
    df['albedo_ratio'] = lunadem.extract_feature_gamma(df)
    df['regolith_depth'] = lunadem.extract_feature_delta(df)
    
    features = [
        'solar_zenith', 'surface_temp', 'elevation', 'slope', 'reflectance',
        'crater_density', 'sensor_noise_alpha', 'sensor_noise_beta',
        'mineral_index', 'thermal_inertia', 'albedo_ratio', 'regolith_depth'
    ]
    
    X = df[features]
    
    print("Loading model from weights_file...")
    model = joblib.load('weights_file/model.pkl')
    
    # Example using another function to evaluate/predict
    predictions = model.predict(X)
    
    print("Sample predictions on live data:")
    for i in range(5):
        # The true label can be verified using predict_label for each row
        # (simulating evaluation logic)
        row = df.iloc[[i]]
        pred = predictions[i]
        
        print(f"Row {i} -> Model Predicted: {pred:.4f}")

if __name__ == "__main__":
    main()
