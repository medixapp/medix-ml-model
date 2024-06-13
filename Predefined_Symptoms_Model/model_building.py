import tensorflow as tf

def build_model(num_classes):
    """
    Build a Keras Sequential model.

    Params:
    - num_classes (int): Number of output classes.

    Returns:
    - model (tf.keras.Model): Uncompiled Keras model.
    """
    tf.keras.backend.clear_session()
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(14, activation='softmax')
    ])
    return model

def compile_model(model, learning_rate=0.0001):
    """
    Compile a Keras model.

    Params:
    - model (tf.keras.Model): The Keras model to compile.
    - learning_rate (float): Learning rate for the optimizer.

    Returns:
    - model (tf.keras.Model): Compiled Keras model.
    """
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model

def train_model(model, x_train, y_train, x_val, y_val, epochs=1000):
    """
    Train a Keras model.

    Params:
    - model (tf.keras.Model): The Keras model to train.
    - x_train (np.ndarray): Training features.
    - y_train (np.ndarray): Training labels.
    - x_val (np.ndarray): Validation features.
    - y_val (np.ndarray): Validation labels.
    - epochs (int): Number of epochs to train.

    Returns:
    - history (tf.keras.callbacks.History): History object containing training history.
    """
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_val, y_val), callbacks=[early_stopping])
    return history

def save_model(model: tf.keras.models.Model) :
    model.save('predefined_model.h5')
