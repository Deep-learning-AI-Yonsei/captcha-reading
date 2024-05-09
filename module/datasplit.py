import numpy as np

def split_data(images, labels, train_size=0.9, shuffle=True):
    # 1. 데이터 셋의 개수
    size = len(images)
    # 2. 데이터 셋의 인덱스를 저장하고 필요시 shuffle
    indices = np.arange(size)
    if shuffle:
        np.random.shuffle(indices)
    # 3. train_size에 맞춰 train set 사이즈 결정
    train_samples = int(size * train_size)
    # 4. train, validation set 나눠주기
    x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
    x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]
    return x_train, x_valid, y_train, y_valid