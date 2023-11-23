# Clasificador de Enfermedades de Roya y Minero en Hojas de Café

## Descripción del Proyecto

Este proyecto se centra en el desarrollo de un clasificador de enfermedades en hojas de café, específicamente Roya y Minero. El conjunto de datos utilizado contiene 2 categorías (Minero y Roya), cada una compuesta por 250 imágenes. Todas las imágenes se redimensionaron a 224 x 224 píxeles para garantizar la consistencia en el conjunto de datos.

## Hiperparámetros a Evaluar

Se evaluaron varios hiperparámetros durante el proceso de entrenamiento del modelo:

- **Modelos a Probar:** Se seleccionaron tres modelos para su evaluación: DenseNet121, EfficientNet-B0 y ResNet101v2.
- **Optimizadores:** Se probaron tres optimizadores diferentes: Adam, SGD y RMSprop.
- **Valores de Learning Rate:** Se consideraron cuatro valores de tasa de aprendizaje: 1e-2, 1e-3, 1e-4 y 1e-5.
- **Batch Size:** Se fijó en 8 para el entrenamiento.
- **Epochs:** Se configuró en 100 durante el entrenamiento del modelo.

## Resultados

Después de evaluar los diferentes combinaciones de modelos, optimizadores y tasas de aprendizaje, se determinó que el mejor conjunto de hiperparámetros fue el siguiente:

- **Mejor Modelo:** ResNet101V2.
- **Optimizador:** SGD.
- **Learning Rate:** 0.0001.

## Entrenamiento del Modelo

El modelo final se entrenó con los siguientes parámetros:

- **Epochs:** 100.
- **Batch Size:** 8.

Después de completar el entrenamiento, el modelo logró un accuracy final de 0.81, lo que indica una capacidad considerable para clasificar eficientemente entre las categorías de Minero y Roya en las hojas de café.
