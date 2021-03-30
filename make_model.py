from tensorflow import keras
# Models & malicious discriminator model
def make_discriminator_model():
  model = keras.Sequential()
  model.add(keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
  model.add(keras.layers.LeakyReLU())
  model.add(keras.layers.Dropout(0.3))

  model.add(keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
  model.add(keras.layers.LeakyReLU())
  model.add(keras.layers.Dropout(0.3))

  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(11))
  return model

# Malicious generator model
def make_generator_model():
  model = keras.Sequential()
    
  model.add(keras.layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
  model.add(keras.layers.BatchNormalization())
  model.add(keras.layers.ReLU())

  model.add(keras.layers.Reshape((7, 7, 256)))
  assert model.output_shape == (None, 7, 7, 256)  # Batch size is not limited

  model.add(keras.layers.Conv2DTranspose(128, (4, 4), strides=(1, 1), padding='same', use_bias=False))
  assert model.output_shape == (None, 7, 7, 128)
  model.add(keras.layers.BatchNormalization())
  model.add(keras.layers.ReLU())

  model.add(keras.layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=False))
  assert model.output_shape == (None, 14, 14, 64)
  model.add(keras.layers.BatchNormalization())
  model.add(keras.layers.ReLU())

  model.add(keras.layers.Conv2DTranspose(1, (4, 4), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
  assert model.output_shape == (None, 28, 28, 1)

  return model