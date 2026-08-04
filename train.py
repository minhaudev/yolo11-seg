# from ultralytics import YOLO

# model = YOLO(r'cfg/models/11/yolo11-seg.yaml')
# model.info()

# model.train(
#         data=r'datasets/apple/data.yaml',
#         imgsz=640,
#         batch=8,                         # Ổn định hơn, ít dao động mAP
#         epochs=2,
#         cache=False,
#         amp=False,                        # FP32 cho độ chính xác cao nhất
#         optimizer='SGD',
#         patience=10,
#         save_period=10,
#         seed=42,
#         project='runs/train',
#         name='exp',
#         workers=0,
#         device='cpu',
#         val=True,
#     )
# #
# # # Load a pretrained YOLO11n model
# # model = YOLO("yolo11n.pt")
# #
# # # Train the model on the COCO8 dataset for 100 epochs
# # train_results = model.train(
# #     data="coco8.yaml",  # Path to dataset configuration file
# #     epochs=100,  # Number of training epochs
# #     imgsz=640,  # Image size for training
# #     device="cpu",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
# # )
# #
# # # Evaluate the model's performance on the validation set
# # metrics = model.val()
# #
# # # Perform object detection on an image
# # results = model("path/to/image.jpg")  # Predict on an image
# # results[0].show()  # Display results
# #
# # # Export the model to ONNX format for deployment
# # path = model.export(format="onnx")  # Returns the path to the exported model


import argparse
import os
import yaml
from ultralytics import YOLO

# ==========================================
# 1. NHẬN THAM SỐ TỪ COMMAND LINE (HỖ TRỢ KAGGLE)
# ==========================================
parser = argparse.ArgumentParser(description="Train YOLO model with custom dataset and milestone")
parser.add_argument('--dataset', type=str, default='BUSI', help="Chọn 1 trong 3: 'BUSBRA', 'BUSI', 'BrEaST'")
parser.add_argument('--milestone', type=int, default=100, help="Mốc số lượng ảnh (VD: 100, 200...)")
parser.add_argument('--epochs', type=int, default=2, help="Số epochs để train (Kaggle nên đặt 100-200)")
parser.add_argument('--device', type=str, default='cpu', help="Thiết bị train: 'cpu' hoặc '0' cho GPU (trên Kaggle dùng '0')")
parser.add_argument('--workers', type=int, default=0, help="Số workers cho dataloader (Kaggle có thể dùng 2 hoặc 4)")
parser.add_argument('--data_dir', type=str, default=None, help="Đường dẫn đến thư mục chứa dữ liệu (Trên Kaggle sẽ là /kaggle/input/...)")
args = parser.parse_args()

TARGET_DATASET = args.dataset
TARGET_MILESTONE = args.milestone

# Lấy đường dẫn thư mục gốc tự động
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Xác định thư mục chứa dữ liệu
if args.data_dir:
    DATASETS_DIR = args.data_dir
else:
    DATASETS_DIR = os.path.join(BASE_DIR, 'datasets', 'FINAL_EXPERIMENTS')

# Từ điển chứa thông tin Classes cho từng bộ
INFO = {
    'BUSBRA': {'nc': 2, 'names': ['benign', 'malignant']},
    'BUSI': {'nc': 2, 'names': ['benign', 'malignant']},
    'BrEaST': {'nc': 1, 'names': ['tumor']}
}

# ==========================================
# 2. TỰ ĐỘNG TẠO FILE YAML THEO ĐÚNG CẤU TRÚC
# ==========================================
dataset_path = os.path.join(DATASETS_DIR, TARGET_DATASET)
train_folder = f'train_{TARGET_MILESTONE}'

# Sửa lỗi cho Kaggle: /kaggle/input là thư mục chỉ đọc (Read-only)
# Nên ta sẽ lưu file yaml tạm thời ở thư mục hiện tại (BASE_DIR) thay vì dataset_path
yaml_path = os.path.join(BASE_DIR, f'data_{TARGET_DATASET}_{TARGET_MILESTONE}.yaml')
yaml_content = {
    'path': dataset_path,
    'train': os.path.join(train_folder, 'images'),
    'val': os.path.join('test_fixed', 'images'),
    'nc': INFO[TARGET_DATASET]['nc'],
    'names': INFO[TARGET_DATASET]['names']
}

# Ghi ra file yaml
with open(yaml_path, 'w', encoding='utf8') as f:
    yaml.dump(yaml_content, f, sort_keys=False)

print(f"Đã tạo cấu hình YAML tại: {yaml_path}")
print(f"BẮT ĐẦU TEST LOCAL: {TARGET_DATASET} - MỐC {TARGET_MILESTONE} ẢNH")

# ==========================================
# 3. HUẤN LUYỆN VỚI CẤU HÌNH LOCAL CỦA BẠN
# ==========================================
# Khởi tạo mô hình (Train từ đầu với kiến trúc yaml như bạn yêu cầu)
model = YOLO(r'cfg/models/11/yolo11-seg.yaml')
model.info()

run_name = f"local_test_{TARGET_DATASET}_{TARGET_MILESTONE}"

# Kế thừa đúng bộ tham số chuẩn của bạn
model.train(
    data=yaml_path,        # Truyền đường dẫn file YAML vừa tạo tự động
    imgsz=640,
    batch=8,               # Ổn định hơn, ít dao động mAP
    epochs=args.epochs,    # Nhận từ dòng lệnh (Kaggle: --epochs 100)
    cache=False,
    amp=False,             # FP32 cho độ chính xác cao nhất
    optimizer='SGD',
    patience=10,
    save_period=10,
    seed=42,
    project='runs/train',
    name=run_name,
    workers=args.workers,  # Nhận từ dòng lệnh (Kaggle: --workers 2)
    device=args.device,    # Nhận từ dòng lệnh (Kaggle: --device 0)
    val=True,
)

print(f"\nTEST LOCAL HOÀN TẤT! Bạn hãy mở thư mục runs/train/{run_name} để xem kết quả nhé.")