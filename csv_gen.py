import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
num_samples = 200000

# Randomly generate feature data
temperature = np.random.uniform(low=10, high=35, size=num_samples)  # Temperature between 10 and 35°C
humidity = np.random.uniform(low=20, high=90, size=num_samples)  # Humidity between 20% and 90%
wind_speed = np.random.uniform(low=0, high=15, size=num_samples)  # Wind speed between 0 and 15 m/s

# Generate air quality based on features
def classify_air_quality(temp, hum, wind):
    if temp > 30 and hum > 70:
        return "Bad"
    elif temp < 15 or wind > 10:
        return "Good"
    else:
        return "Moderate"

air_quality = [classify_air_quality(t, h, w) for t, h, w in zip(temperature, humidity, wind_speed)]

# Create a DataFrame
df = pd.DataFrame({
    'Temperature': temperature,
    'Humidity': humidity,
    'Wind Speed': wind_speed,
    'Air Quality': air_quality
})

# Save DataFrame to CSV
# Update the file path to a valid directory on your Windows system
csv_file_path = r'C:\Users\rdroh\OneDrive\Desktop\Project\air_quality_data.csv'
df.to_csv(csv_file_path, index=False)

print(f"CSV file with 5000 rows of air quality data saved to {csv_file_path}.")
