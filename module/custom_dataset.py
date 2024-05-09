import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from module.transform import resize_with_padding

class CaptchaDataset(Dataset):
    def __init__(self, directory, patch_size=16, transform=None):
        self.directory = directory
        self.patch_size = patch_size  # 패치 크기 설정
        self.transform = transform
        self.images = os.listdir(directory)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.directory, img_name)
        image = Image.open(img_path).convert('RGB')

        # 라벨 추출
        label = img_name.split('.')[0]

        if self.transform:
            image = self.transform(image)

        # 이미지를 패치로 분할
        patches = self.image_to_patches(image, self.patch_size)

        return patches, label
    
    def image_to_patches(self, image, patch_size):
        """
        이미지를 패치로 분할하여 텐서 형태로 변환합니다.
        """
        # 이미지 크기 조정
        image = resize_with_padding(image, (patch_size * 16, patch_size * 16))

        # 이미지 텐서로 변환
        transform = transforms.ToTensor()
        image_tensor = transform(image)

        # 패치로 분할
        _, H, W = image_tensor.shape
        H_patches = H // patch_size
        W_patches = W // patch_size

        patches = image_tensor.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        patches = patches.permute(1, 2, 0, 3, 4).reshape(H_patches * W_patches, -1)

        return patches