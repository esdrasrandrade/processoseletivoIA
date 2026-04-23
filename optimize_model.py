import tensorflow as tf
import os

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # força CPU
from tensorflow import keras


def main():
    modelo_keras = "model.h5"
    modelo_tflite = "model.tflite"

    if not os.path.exists(modelo_keras):
        raise FileNotFoundError(
            f"Arquivo '{modelo_keras}' não encontrado. "
            "Execute primeiro o train_model.py para gerar o modelo treinado."
        )

    # Carrega o modelo treinado
    model = keras.models.load_model(modelo_keras)

    # Converte para TensorFlow Lite com Dynamic Range Quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    # salvando o modelo otimizado
    with open(modelo_tflite, "wb") as f:
        f.write(tflite_model)

    tamanho_h5 = os.path.getsize(modelo_keras) / 1024
    tamanho_tflite = os.path.getsize(modelo_tflite) / 1024

    print(f"Modelo carregado: {modelo_keras}")
    print("Conversão para TensorFlow Lite concluída com Dynamic Range Quantization.")
    print(f"Modelo otimizado salvo em: {modelo_tflite}")
    print(f"Tamanho do model.h5: {tamanho_h5:.2f} KB")
    print(f"Tamanho do model.tflite: {tamanho_tflite:.2f} KB")


if __name__ == "__main__":
    main()