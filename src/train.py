import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.config import (
    X_TRAIN_PATH,
    X_VAL_PATH,
    Y_TRAIN_PATH,
    Y_VAL_PATH,
    MODEL_PATH,
    TRAINING_HISTORY_PATH,
)


X_train = joblib.load(X_TRAIN_PATH)
X_val = joblib.load(X_VAL_PATH)

y_train = joblib.load(Y_TRAIN_PATH)
y_val = joblib.load(Y_VAL_PATH)

print(f"Training features:   {X_train.shape}")
print(f"Validation features: {X_val.shape}")

print(f"Training targets:    {y_train.shape}")
print(f"Validation targets:  {y_val.shape}")



model = keras.Sequential(
    [
        layers.Input(shape=(X_train.shape[1],)),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),

        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),

        layers.Dense(32, activation="relu"),

        layers.Dense(1),
    ]
)



model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),loss="mse",metrics=[keras.metrics.MeanAbsoluteError(name="mae")])


model.summary()


early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True,)

model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=MODEL_PATH, monitor="val_loss", save_best_only=True,)



history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=256,
    callbacks=[
        early_stopping,
        model_checkpoint,
    ],
    verbose=1,
)



joblib.dump(history.history, TRAINING_HISTORY_PATH,)



best_epoch = (min(range(len(history.history["val_loss"])), key=lambda i: history.history["val_loss"][i],) + 1)

print("\nTraining Complete")
print("-----------------")
print(f"Best epoch: {best_epoch}")
print(f"Best validation loss:{min(history.history['val_loss']):,.4f}")

print(f"\nModel saved to:")
print(MODEL_PATH)

print("\nTraining history saved to:")
print(TRAINING_HISTORY_PATH)