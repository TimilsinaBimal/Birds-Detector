import tensorflow as tf
from models.model import get_model
from utils.config import *
from src.data.dataset import Dataset


def callbacks():
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "model.h5", monitor='val_accuracy', mode='max', save_best_only=True)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=3, mode='max', restore_best_weights=True)
    return [model_checkpoint, early_stopping]


def train():
    model = get_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=['accuracy'])
    dataset = Dataset("./")
    train_dataset, val_dataset, test_dataset = dataset.prepare()
    TRAIN_DATA_LEN = dataset.TRAIN_DATA_LEN
    VAL_DATA_LEN = dataset.VAL_DATA_LEN

    history = model.fit(train_dataset, epochs=EPOCHS, steps_per_epoch=TRAIN_DATA_LEN//BATCH_SIZE,
                        validation_data=val_dataset, validation_steps=VAL_DATA_LEN//BATCH_SIZE, callbacks=callbacks())
    return history
