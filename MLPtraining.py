#import necessary modules
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

def main():
    # ........Build model..........
    model = build_model()
    model.compile(loss='mse', optimizer=Adam(learning_rate=1e-4))

    # ..........Prepare data..........
    (trainX, trainY), (valX, valY), (testX, testY), y_mean, y_std = prepare_train_val_test()

    # .......reshape (n,) → (n,1)...........
    trainX = trainX.reshape(-1, 1)
    valX = valX.reshape(-1, 1)
    testX = testX.reshape(-1, 1)

    #........ Train model..........
    model.fit(trainX, trainY, validation_data=(valX, valY),
              epochs=50, batch_size=32, verbose=1)

    # ---- Prediction for smooth curve ----
    X_line = np.linspace(0, 1, 200).reshape(-1, 1)  
    y_pred_scaled = model.predict(X_line)

    # .....Inverse scaling...........
    y_pred = y_pred_scaled * y_std + y_mean
    X_original = X_line * 100.0  
    y_true = my_polynomial(X_original)

    # ---- Plot Original vs Predicted ----
    plt.figure(figsize=(8, 6))
    plt.plot(X_original, y_true, color="blue", label="Original Polynomial", linewidth=2)
    plt.plot(X_original, y_pred, color="red", linestyle="--", label="Predicted Curve", linewidth=2)
    plt.xlabel("X values")
    plt.ylabel("Y values")
    plt.title("Original vs Predicted Curve")
    plt.legend()
    plt.show()

# ---------- Data Preparation ----------
def prepare_train_val_test():
    x, y = data_process()
    total_n = len(x)
    train_n = int(total_n * 0.7)
    val_n = int(total_n * 0.1)
    test_n = int(total_n * 0.2)

    # ......Train/Val/Test split..........
    trainX = x[:train_n]
    trainY = y[:train_n]
    valX = x[train_n: train_n + val_n]
    valY = y[train_n: train_n + val_n]
    testX = x[train_n + val_n:]
    testY = y[train_n + val_n:]

    print('total_n: {}, train_n : {}, val_n: {}, test_n : {}'
          .format(len(x), len(trainX), len(valX), len(testX)))

    # ..........Normalize Y.............
    y_mean = np.mean(trainY)
    y_std = np.std(trainY)
    trainY = (trainY - y_mean) / y_std
    valY = (valY - y_mean) / y_std
    testY = (testY - y_mean) / y_std

    return (trainX, trainY), (valX, valY), (testX, testY), y_mean, y_std

def data_process():
    n = 1000
    x = np.random.randint(0, 100, n)

    # ...........Scale X (0–1 range).........
    x_scaled = x / 100.0

    # ...........Generate polynomial Y.......
    y = []
    for i in range(n):
        y.append(my_polynomial(x[i]))
    y = np.array(y, dtype=np.float32)

    print("Sample x (scaled):", x_scaled[:2])
    print("Sample y:", y[:2])

    return x_scaled, y

def my_polynomial(x):
    return 5*x**2 + 10*x + 5

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

if __name__ == '__main__':
    main()
