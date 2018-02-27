import numpy as np
from PIL import Image
import os
import random


class MnistDataset:

    def __init__(self, data_dir):
        self.cursor = 0

        self.data = []
        self.labels = []

        """
        for i in range(60):
            label = os.path.split(os.path.dirname(data_dir))[i]
            self.labels.append(label)
        print(self.labels)
            
        """

        for i in range(1, 6):
            label = os.path.basename([x[0] for x in os.walk(data_dir)][i])
            self.labels.append(label)

        for label in range(5):
            images = os.listdir(os.path.join(data_dir, self.labels[label]))
            images = [os.path.join(data_dir, self.labels[label], image) for image in images]
            print(self.labels[label])
            for image in images:
                if MnistDataset.isValid(image):
                    self.data.append({'image': image, 'label': label})
        random.shuffle(self.data)

    def sample(self, batch_size):
        batch = self.data[self.cursor:self.cursor+batch_size]
        if len(batch) < batch_size:     # Corner case f we need to wrap around the data
            batch += self.data[0:(self.cursor+batch_size)%len(self.data)]
        self.cursor = (self.cursor + batch_size) % len(self.data)

        images = [sample['image'] for sample in batch]
        images = [MnistDataset.read_image(image) for image in images]
        images = np.stack(images)

        labels = [sample['label'] for sample in batch]
        labels = [MnistDataset.one_hot(label) for label in labels]
        labels = np.stack(labels)

        return images, labels

    def __len__(self):
        return len(self.data)

    @staticmethod
    def isValid(image_path):
        if '.DS_Store' in image_path:
            return False

        return True

    @staticmethod
    def read_image(image_path):
        image = Image.open(image_path)              # Image object
        if image.mode != "RGB":
            image = image.convert('RGB')
        image = image.resize((64,64),Image.ANTIALIAS)
        image = np.array(image)                     # Multi-dimentional array
        image = image.astype(np.float32)            # uint to float
        image = image / 255                         # 0-255 -> 0-1
        # image = image.resize(32, 32)   # (width, height) -> (width, heigh, 1)
        return image

    @staticmethod
    def one_hot(label):
        encoded_label = np.zeros(5)
        encoded_label[label] = 1
        return encoded_label