import gzip
import shutil
import os

# 获取当前脚本所在目录的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))

# 构建完整路径
train_feature_path = os.path.join(script_dir, 'MNIST', 'train-images-idx3-ubyte.gz')
train_label_path = os.path.join(script_dir, 'MNIST', 'train-labels-idx1-ubyte.gz')
test_feature_path = os.path.join(script_dir, 'MNIST', 't10k-images-idx3-ubyte.gz')
test_label_path = os.path.join(script_dir, 'MNIST', 't10k-labels-idx1-ubyte.gz')

save_train_feature_path = os.path.join(script_dir, 'Extracted_MNIST', 'train-images-idx3-ubyte')
save_train_label_path = os.path.join(script_dir, 'Extracted_MNIST', 'train-labels-idx1-ubyte')
save_test_feature_path = os.path.join(script_dir, 'Extracted_MNIST', 't10k-images-idx3-ubyte')
save_test_label_path = os.path.join(script_dir, 'Extracted_MNIST', 't10k-labels-idx1-ubyte')

def extract_gzip_file(gzip_path, extract_path):
    """Extract a .gz file."""
    if not os.path.exists(gzip_path):
        print(f"Error: File {gzip_path} does not exist.")
        return
    try:
        with gzip.open(gzip_path, 'rb') as f_in:
            with open(extract_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"Extracted {gzip_path} to {extract_path}")
    except Exception as e:
        print(f"Failed to extract {gzip_path}: {e}")

if __name__ == "__main__":
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.join(script_dir, 'Extracted_MNIST'), exist_ok=True)

    # 提取所有 gzip 文件
    extract_gzip_file(train_feature_path, save_train_feature_path)
    extract_gzip_file(train_label_path, save_train_label_path)
    extract_gzip_file(test_feature_path, save_test_feature_path)
    extract_gzip_file(test_label_path, save_test_label_path)

    print("MNIST dataset extraction completed.")
