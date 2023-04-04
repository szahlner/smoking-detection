import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
# from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input

from PIL import Image

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from imutils import paths


MODEL_NAME = "cigarette_smoke_detector"
MODEL_EXTENSION = "hdf5"


class CigaretteSmokeDetectorNamespace(argparse.Namespace):
    train_model: bool
    init_lr: float
    epochs: int
    batch_size: int


class CigaretteSmokeDetector:
    def __init__(self, verbose=0):
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.dataset_path = os.path.join(self.path, "dataset")
        self.model_path = os.path.join(self.path, "model", f"{MODEL_NAME}.{MODEL_EXTENSION}")

        if os.path.exists(self.model_path):
            self.detector = load_model(self.model_path)
        else:
            self.detector = None

        self.target_width = 224
        self.target_height = 224

        self.verbose = verbose

    def detect(self, image_url):

        if self.detector is None:
            return None

        # TODO: S3
        image_url = os.path.join(self.path, image_url)
        im_from_folder = Image.open(image_url)
        loaded_image = np.array(im_from_folder.resize((self.target_width, self.target_height), Image.ANTIALIAS))
        loaded_image = np.expand_dims(loaded_image, axis=0)
        prediction = self.detector.predict(preprocess_input(loaded_image), verbose=self.verbose)
        prediction = np.squeeze(prediction)[1]
        return float(prediction)

    def train(self, args):
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        from tensorflow.keras.applications import ResNet50
        # from tensorflow.keras.applications import InceptionResNetV2
        from tensorflow.keras.layers import AveragePooling2D
        from tensorflow.keras.layers import Dropout
        from tensorflow.keras.layers import Flatten
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.layers import Input
        from tensorflow.keras import regularizers
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.preprocessing.image import img_to_array
        from tensorflow.keras.preprocessing.image import load_img
        from tensorflow.keras.utils import to_categorical

        data = []
        labels = []

        if args.debug:
            print("Preparing data and labels")

        image_paths = list(paths.list_images(self.dataset_path))
        for n, image_path in enumerate(image_paths):
            label = image_path.split(os.path.sep)[-2]
            image = load_img(image_path, target_size=(self.target_width, self.target_height))
            image = img_to_array(image)
            image = preprocess_input(image)
            data.append(image)
            labels.append(label)

            if args.debug and (n + 1) % 100 == 0 and n > 0:
                print(f"{n + 1} data and labels prepared")

        data = np.array(data, dtype="float32")

        labels = np.array(labels)
        lb = LabelBinarizer()
        labels = lb.fit_transform(labels)
        labels = to_categorical(labels)

        (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

        augmentation = ImageDataGenerator(
            rotation_range=50,
            zoom_range=0.2,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest",
        )

        input_tensor = Input(shape=(self.target_width, self.target_height, 3))
        base_model = ResNet50(weights="imagenet", include_top=False, input_tensor=input_tensor)
        # base_model = InceptionResNetV2(weights="imagenet", include_top=False, input_tensor=input_tensor)
        
        head_model = base_model.output
        head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
        head_model = Flatten(name="flatten")(head_model)
        head_model = Dense(1024, activation="relu", kernel_regularizer=regularizers.l2(0.002))(head_model)
        head_model = Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.002))(head_model)
        head_model = Dropout(0.5)(head_model)
        head_model = Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.002))(head_model)
        head_model = Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.002))(head_model)
        head_model = Dropout(0.5)(head_model)
        head_model = Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.002))(head_model)
        head_model = Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.002))(head_model)
        head_model = Dropout(0.5)(head_model)
        head_model = Dense(2, activation="softmax")(head_model)
        model = Model(inputs=base_model.input, outputs=head_model)

        for layer in base_model.layers:
            layer.trainable = False

        opt = Adam(lr=args.init_lr, decay=args.init_lr / args.epochs)
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

        if args.debug:
            model.summary()

        history = model.fit(
            augmentation.flow(trainX, trainY, batch_size=args.batch_size),
            steps_per_epoch=len(trainX) // args.batch_size,
            validation_data=(testX, testY),
            validation_steps=len(testX) // args.batch_size,
            epochs=args.epochs
        )

        model.save(self.model_path, save_format="h5")

        if args.debug:
            prediction_idxs = model.predict(testX, batch_size=args.batch_size)
            prediction_idxs = np.argmax(prediction_idxs, axis=1)

            print(classification_report(testY.argmax(axis=1), prediction_idxs, target_names=lb.classes_))

            plt.style.use("ggplot")
            plt.figure()
            plt.plot(np.arange(0, args.epochs), history.history["loss"], label="train_loss")
            plt.plot(np.arange(0, args.epochs), history.history["val_loss"], label="val_loss")
            plt.plot(np.arange(0, args.epochs), history.history["accuracy"], label="train_acc")
            plt.plot(np.arange(0, args.epochs), history.history["val_accuracy"], label="val_acc")
            plt.title("Training Loss and Accuracy")
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend(loc="center right")
            plt.savefig("training.jpg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--init-lr", type=float, default=3e-4, help="initial learning rate")
    parser.add_argument("--epochs", type=int, default=30, help="epochs to train")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size for training")
    parser.add_argument("--debug", action="store_true", help="use debug mode or not")

    args = parser.parse_args(namespace=CigaretteSmokeDetectorNamespace())
    args.debug = True

    csd = CigaretteSmokeDetector()
    csd.train(args)
