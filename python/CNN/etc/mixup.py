import numpy as np


class mixup_generator():
    def __init__(self, x_train, y_train, batch_size=32, alpha=0.2, shuffle=True, datagen=None):
        self.x_train = x_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(x_train)
        self.datagen = datagen

    def __call__(self):
        while True:
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size * 2))

            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                x, y = self.__data_generation(batch_ids)

                yield x, y

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

            return indexes

    def __data_generation(self, batch_ids):
        _, h, w, c = self.x_train.shape
        _, class_num = self.y_train.shape
        x1 = self.x_train[batch_ids[:self.batch_size]]
        x2 = self.x_train[batch_ids[self.batch_size:]]
        y1 = self.y_train[batch_ids[:self.batch_size]]
        y2 = self.y_train[batch_ids[self.batch_size:]]
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        x_1 = l.reshape(self.batch_size, 1, 1, 1)
        y_1 = l.reshape(self.batch_size, 1)

        x = x1 * x_1 + x2 * (1- x_1)
        y = y1 * y_1 + y2 * (1 - y_1)

        if self.datagen:
            for i in range(self.batch_size):
                x[i] = self.datagen.random_transform(x[i])

        return x, y
