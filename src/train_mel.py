from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback

from CRNN import KerasCrnn

K.clear_session()
K.set_image_data_format('channels_last')


class melWeightsSaver(Callback):
    def on_epoch_end(self, epoch, logs={}):
        # Save model architecture
        model_json = mel_crnn.to_json()
        with open('../models/mel.json', 'w') as json_file:
            json_file.write(model_json)

        # Save model weights
        mel_crnn.save_weights('../models/mel.h5')

        print('Saved model')


IMG_HEIGHT = 256
IMG_WIDTH = 256
N_LABELS = 20
BATCH_SIZE = 1
LSTM_SIZE = 32
N_ITER = 10

# Construct CRNN architecture
mel_crnn = KerasCrnn(IMG_HEIGHT, IMG_WIDTH, N_LABELS, nh=LSTM_SIZE)
mel_crnn.compile(loss='categorical_crossentropy',
                 optimizer='rmsprop',
                 metrics=['categorical_accuracy']
                 )

train_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_directory(
    directory='../data/gen/mel/train/',
    # color_mode='rgb',
    color_mode='grayscale',
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=42
)
valid_datagen = ImageDataGenerator()
valid_generator = valid_datagen.flow_from_directory(
    directory='../data/gen/mel/valid/',
    # color_mode='rgb',
    color_mode='grayscale',
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=42
)

model_saver = melWeightsSaver()

STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size

print('\n\n\n\n\n\n\n')
print(mel_crnn.summary())
print('\n\n')
print(f'train_generator.n: {train_generator.n}')
print(f'valid_generator.n: {valid_generator.n}')
print(f'STEP_SIZE_TRAIN: {STEP_SIZE_TRAIN}')
print(f'STEP_SIZE_VALID: {STEP_SIZE_VALID}')
print('\n\n\n\n\n\n\n')

mel_crnn.fit_generator(train_generator,
                       steps_per_epoch=STEP_SIZE_TRAIN,
                       validation_data=valid_generator,
                       validation_steps=STEP_SIZE_VALID,
                       epochs=N_ITER,
                       callbacks=[model_saver]
                       )
