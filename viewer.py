import fiftyone as fo
import fiftyone.zoo as foz

import os

fo.delete_dataset("gs3lam")  # 기존에 생성한 데이터셋이 있다면 삭제
# 데이터 경로
data_dir = "/home/ghryu/Drives/Env-AI/gs3lam-fine-tune/data/Replica/office0"  # 상황에 맞게 변경

# 모든 .png 파일 경로 모으기
image_paths = []
for root, dirs, files in os.walk(data_dir):
    for filename in files:
        if filename.endswith(".png"):
            filepath = os.path.join(root, filename)
            image_paths.append(filepath)

# 새 FiftyOne 데이터셋 생성
dataset = fo.Dataset("gs3lam")  # 원하는 이름 지정

# 이미지 경로들을 Sample로 추가
samples = []
for img_path in image_paths:
    sample = fo.Sample(filepath=img_path)
    samples.append(sample)
    
print(samples)


dataset.add_samples(samples)

# Print a sample ground truth detection
# sample = dataset.first()
# print(sample.predictions.detections[0])

# Open the dataset in the App
session = fo.launch_app(dataset, port=8012, address="0.0.0.0")

session.wait()

