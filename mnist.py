import tensorflow as tf

# load mnist data
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

# build model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss = 'sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# exercise
model.fit(train_images, train_labels, epochs=5)
model.evaluate(test_images, test_labels, verbose=2)

# Display the model's architecture
model.summary()

# save
#tf.keras.models.save_model(model, "my_first_model")
