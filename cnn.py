###### python=3.9

import torch
import torchvision.transforms as transforms
import torch.nn.init
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import time
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


# # 랜덤 시드 고정
# torch.manual_seed(777)

# # GPU 사용 가능일 경우 랜덤 시드 고정
# if device == 'cuda':
#     torch.cuda.manual_seed_all(777)

seed = int(time.time())    
seed = 1708334944

torch.manual_seed(seed)

if device == 'cuda':
    torch.cuda.manual_seed_all(seed)
print(seed)


# 학습에 사용할 하이퍼 파라미터 설정
learning_rate = 0.001
training_epochs = 7
batch_size = 30


# 1-1. 데이터셋 정의
# 커스텀 데이터셋 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_transform=None):
        
        self.classes = sorted(os.listdir(data_dir))  # 클래스 목록
        self.classes.remove('weights')
        self.classes.remove('guideline')
        
       
        self.classes = [item for item in self.classes if 'original' not in item]

        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}  # 클래스를 인덱스로 매핑
        self.images = []  # 이미지 파일 경로
        self.targets = []  # 레이블
        self.transform = transform
        self.target_transform = target_transform

         # 각 클래스의 이미지 파일 경로와 레이블 지정
        for cls_name in self.classes:
            cls_dir = os.path.join(data_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                img_path = os.path.normpath(img_path)  # 플랫폼에 맞게 경로 정규화
                if 'original' not in img_path:
                    self.images.append(img_path)
                    self.targets.append(self.class_to_idx[cls_name])
                
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        self.img_path = img_path
        target = int(self.targets[idx])
        image = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
                   

        return image, target
    
    
# 합성층에 사용될 변환 정의
transform = transforms.Compose([
    transforms.ToTensor()  # 파이토치 텐서로 변환
])
    
    
# 전체 데이터셋을 무작위로 학습용과 테스트용으로 나누기
dataset = CustomDataset(data_dir='./data/', transform=transform)
dataset_size = len(dataset)
train_size = int(dataset_size * 0.8)
validation_size = int(dataset_size * 0.1)
test_size = dataset_size - train_size - validation_size

train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])

# 1-2. 데이터로더로 배치크기 지정
train_data_loader = DataLoader(dataset=train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=True)

validation_data_loader = DataLoader(dataset=validation_dataset,
                    batch_size=3,
                    shuffle=True,
                    drop_last=True)

test_data_loader = DataLoader(dataset=test_dataset,
                    batch_size=5,
                    shuffle=True,
                    drop_last=True)


# 2. 클래스로 모델 설계
class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()    # CNN의 부모 클래스(torch.nn.Module) 호출
        
        # 첫번째층
        # ImgIn shape=(?, 28, 28, 1)
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 두번째층
        # ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 전결합층 7x7x64 inputs -> 10 outputs
        self.fc = torch.nn.Linear(640 // 4 * 480 // 4 * 64, 3, bias=True)

        # 전결합층 한정으로 가중치 초기화 
        torch.nn.init.xavier_uniform_(self.fc.weight)
        
        ### 세이비어 초기화: 가중치 초기화가 모델에 영향을 미침에 따라 초기화 방법 제안.
        ### 방법은 2가지 균등분포 or 정규분포로 초기화. 이전층의 뉴런 개수 / 다음층의 뉴런개수 사용하여 균등분포 범위 정해서 초기화
        ### He 초기화 방법도 있음. 
        ### sigmoid or tanh  사용할 경우 세이비어 초기화 효율적
        ### ReLU 계열 함수 사용할 때는 He 초기화 방법 효율적
        ### ReLU + He 초기화 방법이 좀 더 보편적임

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)   # 전결합층을 위해서 Flatten
        out = self.fc(out)
        self.out = out
        return out


# 3. CNN 모델 정의
model = CNN().to(device) # cuda

# 4. 비용함수와 옵티마이저 정의
criterion = torch.nn.CrossEntropyLoss().to(device)    # 비용 함수에 소프트맥스 함수 포함되어 있음.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 5. 총 배치의 수 출력
total_batch = len(train_data_loader)
print('총 배치의 수 : {}'.format(total_batch))


train_loss_list = []
# 6. 모델 training (시간이 꽤 걸립니다.)
for epoch in range(training_epochs):
    avg_loss = 0

    start = time.time()

    for X, Y in train_data_loader: # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y는 레이블.
        
        # image is already size of (28x28), no reshape
        # label is not one-hot encoded
        X = X.to(device)
        Y = Y.to(device)


        optimizer.zero_grad()
        hypothesis = model(X)
        loss = criterion(hypothesis, Y)    # loss 계산
        loss.backward()                    # 미분
        optimizer.step()                   # 가중치 업데이트
        
        
        # avg_cost += cost / total_batch
        avg_loss += loss.item() / training_epochs  # total_batch가 아니라 len(train_data_loader)로 수정

    train_loss_list.append(avg_loss)

    print('[Epoch: {:>4}] loss = {:>.9} time = {:.4f}'.format(epoch, avg_loss, time.time() - start))


PATH = './data/weights/'


# 손실 비용 그래프
L=train_loss_list
plt.plot(L)
plt.xlabel('Epoch')
plt.ylabel('Average loss')
plt.xticks(range(0, training_epochs))  # X 축에 정수 값만 표시되도록 설정    
plt.savefig(PATH + f'{batch_size}-{training_epochs}')
plt.show()




torch.save(model, PATH + 'model.pt')  # 전체 모델 저장
torch.save(model.state_dict(), PATH + 'model_state_dict.pt')  # 모델 객체의 state_dict 저장
torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict()
}, PATH + 'all.tar')  # 여러 가지 값 저장, 학습 중 진행 상황 저장을 위해 epoch, loss 값 등 일반 scalar값 저장 가능    




# 7. model validation : 모델 검증 
# 계층이나 하이퍼 파라미터의 차이 등으로 인한 성능을 비교
with torch.no_grad():
    model.eval()
    for x, y in validation_data_loader:
        x = x.to(device)
        y = y.to(device)
        
        outputs = model(x)
        # print(f"Outputs : {outputs}")
        # print("--------------------")




# 8. model test > 정확도 : 
# 학습을 진행하지 않을 것이므로 torch.no_grad()
with torch.no_grad():
    # 모델 평가 모드로 전환
    model.eval()
    
    # 정확도 계산을 위해 레이블 저장할 리스트
    all_labels = []
    # 예측 결과 저장할 리스트
    all_predictions = []
    
    # DataLoader를 이용하여 테스트 데이터셋의 배치들에 대해 예측 수행
    for i, (images, labels) in enumerate(test_data_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        
        # 모델로부터 로짓(확률값이 아닌 출력) 계산
        logits = model(images)
        
        
        # 소프트맥스 함수를 사용하여 확률로 변환
        probabilities = F.softmax(logits, dim=1)
        

        # 예측값과 정답 레이블 저장
        all_predictions.extend(torch.argmax(probabilities, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    
    # 리스트를 NumPy 배열로 변환
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    for item, item2 in zip(enumerate(all_predictions), all_labels):
        print(f'{item[0]} : {item[1]} {item2}')
    
    # 정확도 계산
    accuracy = np.mean(all_predictions == all_labels)
    
    # 정확도 출력
    print('Accuracy:', accuracy)