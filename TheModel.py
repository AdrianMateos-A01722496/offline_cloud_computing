import tensorflow as tf

def build_model():
    """CNN sencilla para clasificación de dígitos MNIST."""
    model = tf.keras.Sequential([
        # Primer bloque convolucional: detecta patrones locales simples
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        # Segundo bloque convolucional: detecta patrones más complejos
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        # Capas densas para la clasificación final
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')  # 10 clases (dígitos 0-9)
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
