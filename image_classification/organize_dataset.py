import os
import shutil

def create_category_folders(new_dir):
    # 创建类别文件夹
    os.makedirs(os.path.join(new_dir, 'train', 'cat'), exist_ok=True)
    os.makedirs(os.path.join(new_dir, 'train', 'dog'), exist_ok=True)
    os.makedirs(os.path.join(new_dir, 'val', 'cat'), exist_ok=True)
    os.makedirs(os.path.join(new_dir, 'val', 'dog'), exist_ok=True)

def copy_images_to_folders(base_dir, new_dir):
    # 复制训练集的图片到新文件夹
    for img_file in os.listdir(os.path.join(base_dir, 'train')):
        img_path = os.path.join(base_dir, 'train', img_file)
        if 'cat' in img_file:
            shutil.copy2(img_path, os.path.join(new_dir, 'train', 'cat', img_file))
        elif 'dog' in img_file:
            shutil.copy2(img_path, os.path.join(new_dir, 'train', 'dog', img_file))

    # 复制验证集的图片到新文件夹
    for img_file in os.listdir(os.path.join(base_dir, 'val')):
        img_path = os.path.join(base_dir, 'val', img_file)
        if 'cat' in img_file:
            shutil.copy2(img_path, os.path.join(new_dir, 'val', 'cat', img_file))
        elif 'dog' in img_file:
            shutil.copy2(img_path, os.path.join(new_dir, 'val', 'dog', img_file))

if __name__ == "__main__":
    base_directory = 'data'  # 基础目录
    new_directory = 'new_data'  #新目录
    create_category_folders(base_directory, new_directory)
    copy_images_to_folders(base_directory, new_directory)
    print("文件重组完成。")
