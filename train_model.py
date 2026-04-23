import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # força CPU
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def carregar_dados():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalização para [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Adiciona canal: (28, 28) -> (28, 28, 1)
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    return (x_train, y_train), (x_test, y_test)


def construir_modelo():
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def main():
    # Reprodutibilidade básica
    tf.random.set_seed(42)

    (x_train, y_train), (x_test, y_test) = carregar_dados()
    model = construir_modelo()

    model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Acurácia final no teste: {accuracy:.4f}")

    model.save("model.h5")
    print("Modelo salvo com sucesso em model.h5")


if __name__ == "__main__":
    main()