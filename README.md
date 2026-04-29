# Processo Seletivo – Intensivo Maker | AI

👤 Identificação - **Esdras Rodrigues de Andrade**


# Edge AI - Classificação de Dígitos (MNIST)

##  Descrição do Projeto

Este projeto implementa um pipeline completo de **Edge AI** para classificação de dígitos manuscritos utilizando o dataset MNIST.

O objetivo é desenvolver um modelo leve, eficiente e compatível com dispositivos embarcados, incluindo sua otimização com TensorFlow Lite.

---

## Etapas do Projeto

O pipeline foi dividido em duas etapas principais:

### 1. Treinamento do Modelo
- Utilização de uma CNN (Rede Neural Convolucional)
- Dataset: MNIST (dígitos de 0 a 9)
- Normalização dos dados para melhor desempenho
- Treinamento por 5 épocas
- Salvamento do modelo em formato `.h5`

### 2. Otimização para Edge AI
- Conversão do modelo para TensorFlow Lite
- Aplicação de **Dynamic Range Quantization**
- Redução do tamanho do modelo
- Preparação para execução em dispositivos com baixo poder computacional

---

## Arquitetura do Modelo

A arquitetura foi projetada para ser leve e eficiente:

- Conv2D (32 filtros, ReLU)
- MaxPooling2D
- Conv2D (64 filtros, ReLU)
- MaxPooling2D
- Flatten
- Dropout (0.3)
- Dense (10 classes, Softmax)

---

## 📊 Resultados

- Acurácia no conjunto de teste: **~98%**
- Modelo funcional e otimizado para Edge AI

---

## 📦 Requisitos

```bash
tensorflow>=2.12
numpy
