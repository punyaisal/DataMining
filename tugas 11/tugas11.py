import numpy as np
import pandas as pd

# Dataset
data = {
    "Siswa": [1, 2, 3, 4, 5],
    "DTW": [2, 3, 4, 1, 3],
    "DTT": [4, 4, 3, 5, 2],
    "DMT": [4, 3, 2, 4, 1],
    "DDB": [3, 5, 5, 2, 3],
}

# Convert to DataFrame
df = pd.DataFrame(data).set_index("Siswa")

# Compute Manhattan Distance
def manhattan_distance(df):
    n = len(df)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distances[i, j] = np.sum(np.abs(df.iloc[i] - df.iloc[j]))
    return distances

# Calculate distance matrix
distance_matrix = manhattan_distance(df)
distance_df = pd.DataFrame(
    distance_matrix,
    index=df.index,
    columns=df.index
)

# Print the distance matrix
print("Matriks Jarak (Manhattan Distance):")
print(distance_df)
