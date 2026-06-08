import sys
import pandas as pd
import joblib
import os
import lunadem

def evaluate(csv_path, roll_no):
    # Load test data
    df = pd.read_csv(csv_path)
    
    # Ensure all derived features exist
    if 'mineral_index' not in df.columns:
        df['mineral_index'] = lunadem.extract_feature_alpha(df)
        df['thermal_inertia'] = lunadem.extract_feature_beta(df)
        df['albedo_ratio'] = lunadem.extract_feature_gamma(df)
        df['regolith_depth'] = lunadem.extract_feature_delta(df)
        
    features = [
        'solar_zenith', 'surface_temp', 'elevation', 'slope', 'reflectance',
        'crater_density', 'sensor_noise_alpha', 'sensor_noise_beta',
        'mineral_index', 'thermal_inertia', 'albedo_ratio', 'regolith_depth'
    ]
    
    # Load model
    weights_path = os.path.join('weights_file', 'model.pkl')
    if not os.path.exists(weights_path):
        return
    model = joblib.load(weights_path)
    
    # Predict
    X = df[features]
    predictions = model.predict(X)
    
    # Save to roll_no.csv
    out_df = pd.DataFrame(predictions, columns=['prediction'])
    out_df.to_csv(f"{roll_no}.csv", index=False)
    
    # NOTE: Output NOTHING ELSE! No prints.

if __name__ == "__main__":
    if len(sys.argv) == 3:
        csv_path = sys.argv[1]
        roll_no = sys.argv[2]
        evaluate(csv_path, roll_no)
