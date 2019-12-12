"""
@original author: Viet Nguyen <nhviet1009@gmail.com>
Modified by Houssam Kherraz
"""
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


# from src.config import CLASSES

CLASSES = [str(i) for i in range(1,11)]


class MyDataset(Dataset):
    def __init__(self, root_path="data", total_images_per_class=10000, ratio=0.8, mode="train"):
        self.root_path = root_path
        self.num_classes = len(CLASSES)

        if mode == "train":
            self.offset = 0
            self.num_images_per_class = int(total_images_per_class * ratio)

        else:
            self.offset = int(total_images_per_class * ratio)
            self.num_images_per_class = int(total_images_per_class * (1 - ratio))
        self.num_samples = self.num_images_per_class * self.num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item, dataset='svhn'):
        
        file_ = "{}/full_numpy_bitmap_{}.npy".format(self.root_path, CLASSES[int(item / self.num_images_per_class)])
        image = np.load(file_).astype(np.float32)[self.offset + (item % self.num_images_per_class)]
        if dataset == 'quickdraw':
            image /= 255
            return image.reshape((1, 28, 28)), int(item / self.num_images_per_class)
        else:
            image = np.squeeze(image)
            image = image.astype('uint8')
            # print(type(image))
            # print("SHAPE", image.shape, image[0])
            gray = Image.fromarray(image.T)
            gray = gray.convert('L')
            gray = np.array(gray).astype(np.float32)
            gray /= 255
            return gray.reshape((1, 32, 32)), int(item / self.num_images_per_class)


if __name__ == "__main__":
    training_set = MyDataset("../data", 500, 0.8, "train")
    print(training_set.__getitem__(3))
