import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# Generate simple sine wave
def generate_data():
    t = np.arange(0, 1000, 0.1)
    data = np.sin(0.02 * t)
    return data

# Prepare sequences for a given window size
def create_dataset(series, window_size):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    X = np.array(X)
    y = np.array(y)
    return np.expand_dims(X, axis=-1), y  # [samples, time, features]


# Transformer Block:
from tensorflow.keras import layers, models
import tensorflow as tf

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False, **kwargs):
        attn_output = self.att(inputs, inputs, training=training)
        out1 = self.layernorm1(inputs + self.dropout1(attn_output, training=training))
        ffn_output = self.ffn(out1, training=training)
        return self.layernorm2(out1 + self.dropout2(ffn_output, training=training))
    

# Full Model with Transformer Block
def build_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(64)(inputs)
    x = TransformerBlock(embed_dim=64, num_heads=2, ff_dim=128)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(1)(x)
    return models.Model(inputs, x)


# Experiment Loop:
data = generate_data()

# Window sizes to test
window_sizes = [5, 10, 15, 20, 25, 30, 50, 100, 200]
results = []

for ws in window_sizes:
    print(f"\n🔍 Testing window size: {ws}")
    X, y = create_dataset(data, ws)

    # Train-test split
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    model = build_model((ws, 1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    preds = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    results.append((ws, mse, mape, r2))
    print(f"✅ MAPE for window size {ws}: {mape:.5f}")


# Plot all results
ws_vals, mse_vals, mape_vals, r2_vals = zip(*results)
plt.plot(ws_vals, mse_vals, marker='o', label='MSE')
plt.plot(ws_vals, mape_vals, marker='o', label='MAPE')
plt.plot(ws_vals, r2_vals, marker='o', label='R2')
plt.title("Performance Metrics vs. Window Size")
plt.xlabel("Window Size (Time Lag)")
plt.ylabel("Performance Metric")
plt.legend()
plt.grid(True)
plt.show()

# show results dataframe
import pandas as pd
results_df = pd.DataFrame(results, columns=["Window Size", "MSE", "MAPE", "R2"]).sort_values(by="MAPE")
print("\nResults DataFrame:")
results_df 
