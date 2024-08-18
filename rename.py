import os
import sys

# Đặt bộ mã hóa mặc định thành UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Đường dẫn tới thư mục chứa ảnh
folder_path = 's2'

# Lấy danh sách tất cả các file trong thư mục
files = os.listdir(folder_path)

# Lọc chỉ lấy các file ảnh (ví dụ: .jpg, .png)
image_files = [f for f in files if f.endswith(('.jpg', '.png', '.jpeg'))]

# Sắp xếp danh sách file
image_files.sort()

# Đổi tên các file
for index, filename in enumerate(image_files):
    # Tạo tên file mới
    new_filename = f"{index}.jpg"
    
    # Đường dẫn cũ và mới
    old_file = os.path.join(folder_path, filename)
    new_file = os.path.join(folder_path, new_filename)
    
    # Đổi tên file
    os.rename(old_file, new_file)
    
    print(f"Đã đổi tên: {filename} -> {new_filename}")

print("Hoàn thành việc đổi tên file.")
