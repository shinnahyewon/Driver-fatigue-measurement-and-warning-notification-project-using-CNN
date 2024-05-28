import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 시작 시간 설정
start_time = datetime(2022, 1, 1, 10, 0, 0)

# 데이터 수집 간격 및 기간 설정
time_interval = timedelta(minutes=5)
data_points = 10000

# 눈 깜빡임과 입 크기에 대한 평균과 표준편차를 조절
blink_mean = 3
blink_std = 1
mouth_mean = 0.6
mouth_std = 0.3

timestamps = [start_time + i * time_interval for i in range(data_points)]
blink_counts = np.random.normal(loc=blink_mean, scale=blink_std, size=data_points).astype(int)
mouth_sizes = np.clip(np.random.normal(loc=mouth_mean, scale=mouth_std, size=data_points), 0, 1)

data = {
    'Timestamp': timestamps,
    'BlinkCount': blink_counts,
    'MouthSize': mouth_sizes
}

df = pd.DataFrame(data)

# 데이터 저장 (csv 형식으로)
df.to_csv('./ai/larger_virtual_data.csv', index=False)