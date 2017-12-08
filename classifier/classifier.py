from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf
import glob

class TrafficLightClassifier(object):
    def __init__(self):
        """Traffic Light Classifier"""

        self.input_shape = (600, 800, 3)
        self.num_classes = 3
        self.batch_size = 100
        self.nb_steps = 2*int(2000/self.batch_size)
        self.nb_steps_val = int(550/self.batch_size)
        self.nb_epochs = 20
        self.learning_rate = 0.0001

        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

        pass

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(12, kernel_size=(3, 3), strides=(1, 1),
                         activation='relu',
                         input_shape=self.input_shape))
        model.add(Dropout(0.1))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Conv2D(24, (3, 3), activation='relu'))
        model.add(Dropout(0.1))
        model.add(MaxPooling2D(pool_size=(3, 3)))        
        model.add(Conv2D(48, (3, 3), activation='relu'))
        model.add(Dropout(0.1))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax'))
        self.model = model

        optim = Adam(lr=self.learning_rate)

        self.model.compile(optimizer=optim,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

        self.model.summary()

        pass

    def train_model(self):
        train_datagen = ImageDataGenerator(            
            horizontal_flip=True)
       
        train_generator = train_datagen.flow_from_directory(
            u'../TrafficLight_images_additional',
            target_size=(600, 800),
            batch_size=self.batch_size,
            class_mode='categorical')

        val_generator = train_datagen.flow_from_directory(
            u'../TrafficLight_images',
            target_size=(600, 800),
            batch_size=self.batch_size,
            class_mode='categorical')

        tensorboard_cb = TensorBoard(
            log_dir='./logs', histogram_freq=1,
            batch_size=self.batch_size,
            write_graph=True,
            write_grads=True,
            write_images=True)
        
        self.model.fit_generator(
            train_generator,
            steps_per_epoch=self.nb_steps,
            epochs=10,
            validation_data=val_generator,
            validation_steps=self.nb_steps_val,
            callbacks=[tensorboard_cb])

        pass

    def test_model(self):

        pass

if __name__ == '__main__':
    
    TL = TrafficLightClassifier()
    TL.build_model()
    TL.train_model()
