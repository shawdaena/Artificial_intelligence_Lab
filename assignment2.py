from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

# ---- f(x) ----
def my_polynomial(x):
    return 5*x**2 + 10*x - 2

# ---------- Model ----------
def build_model():
    inputs = Input((1,))
    h2 = Dense(64, activation='relu')(inputs)
    h3 = Dense(128, activation='relu')(h2)
    h4 = Dense(128, activation='relu')(h3)
    h5 = Dense(64, activation='relu')(h4)
    outputs = Dense(1)(h5)
    model = Model(inputs, outputs)
    model.summary(show_trainable=True)
    return model

# ---------- Data ----------
def data_process():
    n = 1000
    x = np.random.uniform(-100, 100, n)   # negative to positive

    # -------compute original Y--------
    y = my_polynomial(x)

    # ------scale X to 0–1-------
    x_min, x_max = x.min(), x.max()
    x_scaled = (x - x_min) / (x_max - x_min)

    # ------scale Y to 0–1--------
    y_min, y_max = y.min(), y.max()
    y_scaled = (y - y_min) / (y_max - y_min)

    return x_scaled, y_scaled, x_min, x_max, y_min, y_max

def prepare_train_val_test():
    x_scaled, y_scaled, x_min, x_max, y_min, y_max = data_process()
    total_n = len(x_scaled)
    train_n = int(total_n * 0.7)
    val_n = int(total_n * 0.1)
    test_n = total_n - train_n - val_n

    trainX = x_scaled[:train_n].reshape(-1, 1)
    trainY = y_scaled[:train_n]
    valX = x_scaled[train_n: train_n + val_n].reshape(-1, 1)
    valY = y_scaled[train_n: train_n + val_n]
    testX = x_scaled[train_n + val_n:].reshape(-1, 1)
    testY = y_scaled[train_n + val_n:]

    return (trainX, trainY), (valX, valY), (testX, testY), x_min, x_max, y_min, y_max

# ---------- Main ----------
def main():
    model = build_model()
    model.compile(loss='mse', optimizer=Adam(learning_rate=1e-4))

    (trainX, trainY), (valX, valY), (testX, testY), x_min, x_max, y_min, y_max = prepare_train_val_test()

    model.fit(trainX, trainY, validation_data=(valX, valY), epochs=100, batch_size=32, verbose=1)

    # ---- Prediction ----
    X_line = np.linspace(-100, 100, 100).reshape(-1, 1)
    X_line_scaled = (X_line - x_min) / (x_max - x_min)
    y_pred_scaled = model.predict(X_line_scaled)

    # -----Original Y scaled------
    y_true_scaled = (my_polynomial(X_line) - y_min) / (y_max - y_min)

    # ---- Plot ----
    plt.figure(figsize=(8,6))
    plt.plot(X_line, y_true_scaled, color="blue", label="Original f(x) (0–1)", linewidth=2)
    plt.plot(X_line, y_pred_scaled, color="red", linestyle="--", label="Predicted f(x) (0–1)", linewidth=2)
    plt.xlabel("X values")
    plt.ylabel("Y values")
    plt.title("Original vs Predicted f(x) (Normalized)")
    plt.grid()
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
