from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import numpy as np
import cv2
import glob


def mask_color(imgin):
    # z, l, h = 0, 240, 255
    # red_range = ((l, z, z), (h, h, h))
    # ylw_range = ((l, l, z), (h, h, h))
    # grn_range = ((z, l, z), (h, h, h))
    #
    # # Generate a mash
    # rmask = cv2.inRange(imgin, red_range[0], red_range[1])
    # ymask = cv2.inRange(imgin, ylw_range[0], ylw_range[1])
    # gmask = cv2.inRange(imgin, grn_range[0], grn_range[1])
    #
    # # Apply the mask
    # rimg = cv2.bitwise_and(imgin, imgin, mask=rmask)
    # yimg = cv2.bitwise_and(imgin, imgin, mask=ymask)
    # gimg = cv2.bitwise_and(imgin, imgin, mask=gmask)
    #
    # # Combine masked images and return
    # imgout = cv2.bitwise_or(rimg, yimg)
    # imgout = cv2.bitwise_or(imgout, gimg)
    return imgin


def test_mask():
    wait_time = 500

    img = cv2.imread('../TrafficLight_images/red/cam_img-0000.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgout = mask_color(img)
    cv2.imshow('RED', imgout)

    cv2.waitKey(wait_time)
    img = cv2.imread('../TrafficLight_images/ylw/cam_img-0325.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgout = mask_color(img)
    cv2.imshow('YLW', imgout)

    cv2.waitKey(wait_time)

    img = cv2.imread('../TrafficLight_images/grn/cam_img-0230.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgout = mask_color(img)
    cv2.imshow('GRN', imgout)
    cv2.waitKey(wait_time)

    return

class TrafficLightClassifier(object):
    def __init__(self):
        """Traffic Light Classifier"""

        self.input_shape = (600, 800, 3)
        self.num_classes = 3
        self.batch_size = 25
        self.nb_steps = int(4000/self.batch_size)
        self.nb_steps_val = int(550/self.batch_size)
        self.nb_epochs = 1
        self.learning_rate = 0.0001

        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

        pass

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(12, kernel_size=(5, 5), strides=(1, 1),
                         activation='relu',
                         input_shape=self.input_shape))
        model.add(Dropout(0.1))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(24, (5, 5), activation='relu'))
        model.add(Dropout(0.1))
        model.add(MaxPooling2D(pool_size=(3, 3)))        
        model.add(Conv2D(48, (5, 5), activation='relu'))
        model.add(Dropout(0.1))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Flatten())
        #model.add(Dense(100, activation='relu'))
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
            preprocessing_function=mask_color,
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
            epochs=self.nb_epochs,
            validation_data=val_generator,
            validation_steps=self.nb_steps_val,
            callbacks=[tensorboard_cb])

        pass

    def get_data(self):
        xdata = []
        ydata = []
        max_count = 30
        red_files = glob.glob(r'../TrafficLight_images/red/*.png')
        count = 0
        for fil in red_files:
            if count > max_count:
                break
            img = cv2.imread(fil)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgout = mask_color(img)
            xdata.append(imgout)
            ydata.append((1.0, 0.0, 0.0))
            count += 1

        grn_files = glob.glob(r'../TrafficLight_images/grn/*.png')
        count = 0
        for fil in grn_files:
            if count > max_count:
                break
            img = cv2.imread(fil)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgout = mask_color(img)
            xdata.append(imgout)
            ydata.append((0.0, 1.0, 0.0))
            count += 1

        ylw_files = glob.glob(r'../TrafficLight_images/ylw/*.png')
        count = 0
        for fil in ylw_files:
            if count > max_count:
                break
            img = cv2.imread(fil)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgout = mask_color(img)
            xdata.append(imgout)
            ydata.append((0.0, 0.0, 1.0))
            count += 1

        self.x_test = xdata
        self.y_test = ydata
        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        #     xdata, ydata, test_size=0.2, random_state=42)

    def test_model(self):
        labels = ['red   ', 'yellow', 'green ']
        for img, lbl in zip(self.x_test, self.y_test):
            img1 = img.reshape(1,600,800,3)
            pred = self.model.predict(img1, batch_size=1)
            target = labels[np.argmax(lbl)]
            result = labels[np.argmax(pred)]
            if result==target:
                check = "."
            else:
                check = "** FAIL **"
            print("{} | RED = {:9.3f} YLW = {:9.3f} GRN = {:9.3f} | {}".format(target, pred[0][0], pred[0][1], pred[0][2], check))
        pass

if __name__ == '__main__':
    
    TL = TrafficLightClassifier()
    TL.get_data()
    TL.build_model()
    TL.train_model()
    TL.test_model()
    # test_mask()
