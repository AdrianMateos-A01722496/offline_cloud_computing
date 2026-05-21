# Aprendizaje Federado — MNIST

Simulación de aprendizaje federado con el dataset MNIST. Cada miembro del equipo entrena un modelo CNN localmente sobre su partición de datos, y luego se computan modelos globales usando tres métodos de agregación: FedAvg, FedMedian y FedAcc.

## Estructura

| Archivo | Descripción |
|---|---|
| `TheModel.py` | Arquitectura CNN compartida por todos los miembros |
| `training_local.ipynb` | Entrenamiento local + curvas de aprendizaje + classification report |
| `global_model.ipynb` | Cómputo del modelo global con FedAvg, FedMedian y FedAcc |
| `data_split.py` | División del dataset (**no está en el repo**, cada quien lo corre localmente) |

## Cómo correr

### 1. Instalar dependencias
```bash
pip install tensorflow scikit-learn matplotlib numpy
```
O con `uv`:
```bash
uv sync
```

### 2. Generar tu partición local
Edita `data_split.py` y cambia `MEMBER_ID` a tu número (0 = Adrian, 1 = Rogelio, 2 = Tony), luego:
```bash
python data_split.py
```
Esto genera `x_local.npy` y `y_local.npy` — no los subas al repo.

### 3. Entrenar localmente
Abre `training_local.ipynb`, confirma tu `MEMBER_ID` y ejecuta todas las celdas.
Al terminar tendrás un archivo `weights_N.npz` — compártelo con el equipo por fuera del repo.

### 4. Calcular el modelo global
El miembro que reúna los tres archivos `weights_0.npz`, `weights_1.npz` y `weights_2.npz` ejecuta `global_model.ipynb`.

## Métodos de agregación

- **FedAvg** — promedio ponderado de pesos por número de muestras (baseline)
- **FedMedian** — mediana coordenada a coordenada, más robusto ante datos ruidosos
- **FedAcc** — promedio ponderado por la precisión local de cada modelo en el test set global
