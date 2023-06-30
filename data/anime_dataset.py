import glob
import numpy as np
from torchvision import io
from torch.utils.data import Dataset

DEFAULT_FILE_EXTENSION = ['*.jpg', '*.png', '*.jpg', '*.png', '*jpg']


class AnimeDataset(Dataset):
    def __init__(self, data_dir, mode, file_extentions=DEFAULT_FILE_EXTENSION, transformer=None):
        super(Dataset, self).__init__()

        if type(data_dir) is not list:
            raise Exception('data_dir should be list')

        self.mode = mode
        self.data_dir = data_dir
        self.transforms = transformer
        self.file_extentions = file_extentions

        self.image_list = self.__get_img_list()
        self.img_len = len(self.image_list)

    def __len__(self):
        return self.img_len

    def __get__img_filenames(self, image_list):
        validation_size = int(np.ceil(len(image_list) * 0.15))
        match self.mode:
            case "train":
                return image_list[2 * validation_size:]

            case "val":
                return image_list[validation_size: 2 * validation_size]

            case "test":
                return image_list[:validation_size]

            case _:
                raise Exception('Unknown mode')

    def __get_img_list(self):
        image_list = []

        for index in range(len(self.data_dir)):
            total_files = glob.glob(
                self.data_dir[index] + self.file_extentions[index])
            for filename in self.__get__img_filenames(total_files):
                image_list.append(filename)
            print(len(total_files))

        return image_list

    def __getitem__(self, index):
        image = io.read_image(
            self.image_list[index], mode=io.ImageReadMode.RGB)
        # image = (image - 127.5) / 127.5

        if self.transforms:
            image = self.transforms(image)

        return image


if __name__ == '__main__':
    import matplotlib.pyplot as plt # we just need while we run this script directly

    test_data = AnimeDataset(mode='train', data_dir=['./data/images/'])

    print(test_data[0].shape)
    plt.figure(figsize=(8, 8))
    plt.imshow(test_data[0].permute(1, 2, 0))
    plt.show()
