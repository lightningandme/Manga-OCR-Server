import cv2
import numpy as np
print(f"OpenCV 版本: {cv2.__version__}")
print(f"Numpy 版本: {np.__version__}")

# 尝试创建一个黑色的空矩阵
test_img = np.zeros((100, 100, 3), dtype=np.uint8)
print("环境检查通过，可以开始编写智能截取逻辑了！")