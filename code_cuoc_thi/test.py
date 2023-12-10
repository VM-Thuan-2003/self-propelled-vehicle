import torch
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device

# Chọn thiết bị GPU nếu có, nếu không sẽ chọn CPU
device = select_device('0' if torch.cuda.is_available() else 'cpu')

# Load mô hình YOLOv5 đã được huấn luyện
weights_path = 'E:\\WORKPACE\\XE_TU_HANH\\labelme\\yolov5-master\\yolov5-master\\runs\\train\\ModelSignNew2\\weights\\best.pt'
model = attempt_load(weights_path, map_location=device)

# Đặt mô hình vào chế độ đánh giá
model.eval()

# Load ảnh cần detect
image_path = 'D:\\DOWNLOADS\\via-trafficsign\\images\\train\\00247.jpg'
img = Image.open(image_path)

# Chuyển ảnh thành tensor
img = torch.from_numpy(np.array(img)).to(device)

# Thực hiện detect
results = model(img.unsqueeze(0))[0]

# Lọc kết quả bằng non-maximum suppression
results = non_max_suppression(results['xyxy'].cpu(), conf_thres=0.5, iou_thres=0.5)

# Vẽ bounding boxes lên ảnh
if results[0] is not None:
    for box in results[0]:
        # box = scale_coords(img.shape[2:], box[1:], img.shape).round()
        box = [int(i) for i in box]
        img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

# Hiển thị ảnh kết quả
plt.imshow(img)
plt.show()
