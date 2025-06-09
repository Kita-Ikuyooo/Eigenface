import os
import sys
import cv2
import numpy as np
from typing import List, Tuple, Dict


# 通用图像处理函数
def load_image_and_eyes(image_path: str) -> Tuple[np.ndarray, List[float]]:
    """加载图像和对应的眼睛位置"""
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")

    # 读取眼睛位置文件
    base_name = os.path.splitext(image_path)[0]
    txt_path = base_name + '.txt'

    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"找不到眼睛位置文件: {txt_path}")

    with open(txt_path, 'r') as f:
        eye_data = f.readline().split()
        if len(eye_data) < 4:
            raise ValueError(f"无效的眼睛位置文件: {txt_path}")
        x1, y1, x2, y2 = map(float, eye_data[:4])

    return img, [x1, y1, x2, y2]


def align_face(img: np.ndarray, eyes: List[float],
               target_size: Tuple[int, int] = (100, 100)) -> np.ndarray:
    """根据眼睛位置对齐人脸"""
    # 提取眼睛坐标
    left_eye = (eyes[0], eyes[1])
    right_eye = (eyes[2], eyes[3])

    # 计算眼睛中心点和角度
    eye_center = ((left_eye[0] + right_eye[0]) / 2,
                  (left_eye[1] + right_eye[1]) / 2)
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    # 目标眼睛位置（标准化后）
    target_left = (target_size[0] * 0.35, target_size[1] * 0.5)
    target_right = (target_size[0] * 0.65, target_size[1] * 0.5)
    target_center = ((target_left[0] + target_right[0]) / 2,
                     (target_left[1] + target_right[1]) / 2)

    # 计算缩放比例
    dist = np.sqrt(dx ** 2 + dy ** 2)
    target_dist = target_right[0] - target_left[0]
    scale = target_dist / dist

    # 构建变换矩阵
    rot_mat = cv2.getRotationMatrix2D(eye_center, angle, scale)

    # 调整平移量
    rot_mat[0, 2] += target_center[0] - eye_center[0]
    rot_mat[1, 2] += target_center[1] - eye_center[1]

    # 应用仿射变换
    aligned = cv2.warpAffine(img, rot_mat, target_size,
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    return aligned


# 训练程序
def train(energy_percent: float, model_file: str, train_dir: str):
    """训练Eigenface模型"""
    # 收集训练数据
    faces = []
    labels = []
    label_names = []

    # 遍历训练目录
    for person_id, person_name in enumerate(sorted(os.listdir(train_dir))):
        person_dir = os.path.join(train_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        label_names.append(person_name)

        # 遍历每个人的图像
        for file_name in sorted(os.listdir(person_dir)):
            if file_name.lower().endswith('.pgm'):
                image_path = os.path.join(person_dir, file_name)

                try:
                    img, eyes = load_image_and_eyes(image_path)
                    aligned = align_face(img, eyes)
                    faces.append(aligned.flatten())
                    labels.append(person_id)
                except Exception as e:
                    print(f"处理图像时出错 {image_path}: {str(e)}")

    if not faces:
        raise ValueError("未找到有效的训练数据")

    # 转换为NumPy数组
    X = np.array(faces, dtype=np.float64).T  # 每列是一个样本
    y = np.array(labels)

    # 计算平均脸
    mean_face = np.mean(X, axis=1)
    X_centered = X - mean_face[:, np.newaxis]

    # 计算协方差矩阵（使用高效方法）
    n_samples = X_centered.shape[1]
    cov_matrix = np.dot(X_centered.T, X_centered) / n_samples

    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # 按特征值降序排序
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 转换为原始空间的特征向量
    eigenvectors = np.dot(X_centered, eigenvectors)

    # 归一化特征向量
    for i in range(eigenvectors.shape[1]):
        eigenvectors[:, i] /= np.linalg.norm(eigenvectors[:, i])

    # 根据能量百分比选择特征向量
    total_energy = np.sum(eigenvalues)
    cumulative_energy = np.cumsum(eigenvalues) / total_energy
    k = np.argmax(cumulative_energy >= energy_percent) + 1

    # 确保k不超过特征向量数量
    k = min(k, eigenvectors.shape[1])

    # 选择前k个特征向量
    eigenvectors = eigenvectors[:, :k]
    eigenvalues = eigenvalues[:k]

    # 将训练数据投影到特征空间
    projections = np.dot(eigenvectors.T, X_centered)

    # 保存模型
    np.savez(
        model_file,
        mean_face=mean_face,
        eigenvectors=eigenvectors,
        eigenvalues=eigenvalues,
        projections=projections,
        labels=y,
        label_names=label_names
    )

    # 显示前10个特征脸
    show_eigenfaces(eigenvectors, (100, 100))
    print(f"训练完成! 使用 {k} 个特征向量 (能量: {cumulative_energy[k - 1] * 100:.2f}%)")
    print(f"模型已保存到 {model_file}")


def show_eigenfaces(eigenvectors: np.ndarray, face_size: Tuple[int, int]):
    """显示前10个特征脸"""
    # 只取前10个
    n = min(10, eigenvectors.shape[1])
    eigenfaces = eigenvectors[:, :n].copy()

    # 创建大图像用于显示
    rows = 2
    cols = 5
    big_image = np.zeros((rows * face_size[0], cols * face_size[1]), dtype=np.uint8)

    for i in range(n):
        # 获取并归一化特征脸
        eig_face = eigenfaces[:, i].reshape(face_size)
        eig_face = cv2.normalize(eig_face, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # 放置到大图像中
        row = i // cols
        col = i % cols
        big_image[row * face_size[0]:(row + 1) * face_size[0],
        col * face_size[1]:(col + 1) * face_size[1]] = eig_face

    # 显示
    cv2.imshow('Top 10 Eigenfaces', big_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 识别程序
def test(image_path: str, model_file: str):
    """使用训练好的模型识别人脸"""
    # 加载模型
    data = np.load(model_file, allow_pickle=True)
    mean_face = data['mean_face']
    eigenvectors = data['eigenvectors']
    projections = data['projections']
    labels = data['labels']
    label_names = data['label_names']

    # 加载并处理测试图像
    img, eyes = load_image_and_eyes(image_path)
    aligned = align_face(img, eyes)
    test_face = aligned.flatten().astype(np.float64)

    # 减去平均脸
    test_face_centered = test_face - mean_face

    # 投影到特征空间
    test_projection = np.dot(eigenvectors.T, test_face_centered)

    # 计算与所有训练样本的距离
    distances = np.linalg.norm(projections - test_projection[:, np.newaxis], axis=0)

    # 找到最近邻
    min_idx = np.argmin(distances)
    predicted_label = labels[min_idx]
    predicted_name = label_names[predicted_label]

    # 显示结果
    show_results(img, aligned, predicted_name, min_idx, data)


def show_results(original: np.ndarray, aligned: np.ndarray,
                 name: str, min_idx: int, model_data: dict):
    """显示识别结果"""
    # 在原始图像上显示识别结果
    original_display = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    cv2.putText(original_display, f"Identity: {name}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 获取最相似的训练图像
    face_size = (100, 100)
    train_faces = model_data['projections'].shape[1]
    if min_idx < train_faces:
        # 从模型中重建最相似人脸
        eig_vecs = model_data['eigenvectors']
        mean_face = model_data['mean_face']
        proj = model_data['projections'][:, min_idx]
        reconstructed = np.dot(eig_vecs, proj) + mean_face
        similar_face = reconstructed.reshape(face_size)
        similar_face = cv2.normalize(similar_face, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    else:
        similar_face = np.zeros(face_size, dtype=np.uint8)

    # 显示图像
    cv2.imshow('Test Image', original_display)
    cv2.imshow('Aligned Face', aligned)
    cv2.imshow('Most Similar Face', similar_face)

    print(f"识别结果: {name}")
    print("按任意键退出...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 主程序
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("请指定模式: train 或 test")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "train":
        if len(sys.argv) < 4:
            print("用法: python eigenface.py train <能量百分比> <模型文件> [训练目录]")
            sys.exit(1)

        energy_percent = float(sys.argv[2])
        model_file = sys.argv[3]
        train_dir = sys.argv[4] if len(sys.argv) > 4 else "train"

        train(energy_percent, model_file, train_dir)

    elif mode == "test":
        if len(sys.argv) < 4:
            print("用法: python eigenface.py test <测试图像> <模型文件>")
            sys.exit(1)

        test_image = sys.argv[2]
        model_file = sys.argv[3]

        test(test_image, model_file)

    else:
        print("无效的模式. 请使用 'train' 或 'test'")