import os
from PIL import Image
from torch.utils.data import Dataset

class CaptchaDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.images = os.listdir(directory)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.directory, img_name)
        image = Image.open(img_path).convert('RGB')  # RGB 변환 추가
        
        # 이미지 라벨 추출 (파일 이름에서 숫자 추출을 문자열 추출로 변경)
        label = img_name.split('.')[0]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label