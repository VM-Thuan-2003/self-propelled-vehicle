import cv2
import numpy as np

# Đọc ảnh và chuyển sang ảnh xám
image = cv2.imread('./imgs/lane_treeShadow.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Làm mờ và áp dụng phân ngưỡng để tạo ảnh nhị phân
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

# Tìm contours và vẽ contours lên ảnh
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

# Áp dụng HoughLines để tìm các đường thẳng
lines = cv2.HoughLines(thresh, 1, np.pi / 180, threshold=100)

# Vẽ đường thẳng lên ảnh
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Tạo mặt nạ cho các contours
mask = np.zeros_like(gray)
cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

# Tạo mặt nạ cho đường thẳng
for line in lines:
    x1, y1, x2, y2 = cv2.fitLine(line, cv2.DIST_L2, 0, 0.01, 0.01)
    cv2.line(mask, (int(x1 - 1000 * x2), int(y1 - 1000 * y2)), (int(x1 + 1000 * x2), int(y1 + 1000 * y2)), 255, 2)

# Áp dụng bitwise_and giữa ảnh gốc và mặt nạ
result = cv2.bitwise_and(image, image, mask=mask)

# Hiển thị ảnh gốc và kết quả
cv2.imshow('Original Image', image)
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
