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


import os

import yaml

from ultralytics import YOLO

# ==========================================
# 1. BIẾN ĐIỀU KHIỂN (THAY ĐỔI TRƯỚC KHI TEST)
# ==========================================
TARGET_DATASET = "BUSI"  # Chọn 1 trong 3: 'BUSBRA', 'BUSI', 'BrEaST'
TARGET_MILESTONE = 100  # Đặt mốc test thử (VD: 100, 200...)

# Lấy đường dẫn thư mục gốc tự động
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Đảm bảo thư mục FINAL_EXPERIMENTS nằm cùng cấp với file code này
# (Đã sửa lại để trỏ đúng vào thư mục 'datasets/FINAL_EXPERIMENTS' nơi chứa ảnh)
DATASETS_DIR = os.path.join(BASE_DIR, "datasets", "FINAL_EXPERIMENTS")

# Từ điển chứa thông tin Classes cho từng bộ
INFO = {
    "BUSBRA": {"nc": 2, "names": ["benign", "malignant"]},
    "BUSI": {"nc": 2, "names": ["benign", "malignant"]},
    "BrEaST": {"nc": 1, "names": ["tumor"]},
}

# ==========================================
# 2. TỰ ĐỘNG TẠO FILE YAML THEO ĐÚNG CẤU TRÚC
# ==========================================
dataset_path = os.path.join(DATASETS_DIR, TARGET_DATASET)
train_folder = f"train_{TARGET_MILESTONE}"

yaml_path = os.path.join(dataset_path, f"data_{TARGET_MILESTONE}.yaml")
yaml_content = {
    "path": dataset_path,
    "train": os.path.join(train_folder, "images"),
    "val": os.path.join("test_fixed", "images"),
    "nc": INFO[TARGET_DATASET]["nc"],
    "names": INFO[TARGET_DATASET]["names"],
}

# Ghi ra file yaml
os.makedirs(dataset_path, exist_ok=True)
with open(yaml_path, "w", encoding="utf8") as f:
    yaml.dump(yaml_content, f, sort_keys=False)

print(f"Đã tạo cấu hình YAML tại: {yaml_path}")
print(f"BẮT ĐẦU TEST LOCAL: {TARGET_DATASET} - MỐC {TARGET_MILESTONE} ẢNH")

# ==========================================
# 3. HUẤN LUYỆN VỚI CẤU HÌNH LOCAL CỦA BẠN
# ==========================================
# Khởi tạo mô hình (Train từ đầu với kiến trúc yaml như bạn yêu cầu)
model = YOLO(r"cfg/models/11/yolo11-seg.yaml")
model.info()

run_name = f"local_test_{TARGET_DATASET}_{TARGET_MILESTONE}"

# Kế thừa đúng bộ than số chuẩn của bạn
model.train(
    data=yaml_path,  # Truyền đường dẫn file YAML vừa tạo tự động
    imgsz=640,
    batch=8,  # Ổn định hơn, ít dao động mAP
    epochs=2,  # Chạy 2 epochs để test luồng dữ liệu (Pipeline)
    cache=False,
    amp=False,  # FP32 cho độ chính xác cao nhất
    optimizer="SGD",
    patience=10,
    save_period=10,
    seed=42,
    project="runs/train",
    name=run_name,
    workers=0,  # Windows bắt buộc để 0 để tránh lỗi đa luồng (multiprocessing)
    device="cpu",  # Test nhẹ bằng CPU
    val=True,
)

print(f"\nTEST LOCAL HOÀN TẤT! Bạn hãy mở thư mục runs/train/{run_name} để xem kết quả nhé.")
