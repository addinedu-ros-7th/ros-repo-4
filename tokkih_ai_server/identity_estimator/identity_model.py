import torch
import torch.nn as nn
from torchvision.models import resnet18

class AgeNormalizer:
    """ 나이 값을 0~1 범위로 정규화 및 역변환 """
    def __init__(self, min_age=0, max_age=100):
        self.min_age = min_age
        self.max_age = max_age

    def normalize(self, age):
        return (age - self.min_age) / (self.max_age - self.min_age)

    def denormalize(self, normalized_age):
        return normalized_age * (self.max_age - self.min_age) + self.min_age

class AgeLoss(nn.Module):
    """ 나이 추정 손실 함수: MSE + 절대 오차 """
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        diff_loss = torch.mean(torch.abs(pred - target))
        return mse_loss + self.alpha * diff_loss

class ImprovedGenderAgeModel(nn.Module):
    """ ResNet18 기반 성별/나이 추정 모델 """
    def __init__(self, pretrained=True):
        super(ImprovedGenderAgeModel, self).__init__()
        self.backbone = resnet18(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.shared_fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(512)
        )

        self.age_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 나이를 [0,1]로 정규화
        )

        self.gender_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(256),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.shared_fc(x)
        age = self.age_head(x)
        gender = self.gender_head(x)
        return age, gender
