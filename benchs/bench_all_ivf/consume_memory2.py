import gc
import time

# 16GB를 20번에 나누어 할당할 크기 계산
# float는 8바이트이므로 8GB / 10 = 0.8GB, 따라서 각 리스트에 들어갈 요소의 수는 다음과 같습니다.
num_elements = (8 * 1024 ** 3 // 8) // 10  # 약 0.8GB 요소

arrays = []


for i in range(20):
    # 0.8GB 메모리 할당
    large_array = [0.0] * num_elements
    arrays.append(large_array)

    # 리스트에 접근하여 첫 번째와 마지막 요소를 설정하고 출력
    large_array[0] = i + 1.0
    large_array[-1] = (i + 1) * 10.0

print("모든 메모리 할당 및 접근 완료.")

# 1시간 동안 대기 (3600초)
time.sleep(3600)

