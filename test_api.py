import requests
from src.data.load_data import load_wisconsin_dataset
from collections import Counter

wisconsin_tuple = load_wisconsin_dataset()
df = wisconsin_tuple[0]
data = df.to_dict(orient="records")

response = requests.post("http://127.0.0.1:8000/predict", json=data).json()


print("\n ===== BATCH PREDICTION SUMMARY =====")

print(f"Number of samples processed: {response['n_samples']}")

print(f"   Raw classes: {Counter(response['predictions'])}")
print(f"   Labels:      {Counter(response['labels'])}")

print(f"   Total latency: {response['latency_total_seconds']:.6f} seconds")
print(f"   Latency per sample: {response['latency_per_sample_seconds']:.6f} seconds")
