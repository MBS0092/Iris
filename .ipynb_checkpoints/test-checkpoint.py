import tensorflow as tf

model = tf.keras.models.load_model(
    "iris_unet.h5",
    compile=False
)

print("MODEL LOADED")