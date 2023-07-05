import cv2
from PIL import Image
import numpy as np


def mask_to_points(mask:Image):
    """
    Convert a mask to corner points
    :param mask: mask image (PIL格式)
    :return: list of points : left_top, right_top, left_bottom, right_bottom # width * height
    """
    
    cv_mask = cv2.cvtColor(np.asarray(mask), cv2.COLOR_RGB2BGR)
    height, width, _ = cv_mask.shape
    scale = 300
    # 形态学处理除去面积过小的区域
    kernel_size = (int(width / scale), int(height / scale))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    # 开运算
    cv_mask = cv2.morphologyEx(cv_mask, cv2.MORPH_OPEN, kernel)

    # 找出其中最大的边界区域
    gray = cv2.cvtColor(cv_mask, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    max_contour = max(contours, key=cv2.contourArea)

    # 进行多边形逼近
    if max_contour is None:
        return None
    epsilon = 0.02 * cv2.arcLength(max_contour, True)
    approx1 = cv2.approxPolyDP(max_contour, epsilon, True) # [num * height * width]
    # cv2.polylines(cv_mask, [approx1], True, (255, 0, 0), 2)

    # print(len(approx1))  # 角点的个数
    # cv2.imwrite('approxPloyDP.png', cv_mask)

    # 遍历坐标点，找到边界四点
    # if len(approx1) != 4:
    #     return None
    approx1 = approx1.reshape(-1, 2)
    sum = approx1.sum(axis=1)
    diff = np.diff(approx1, axis=1)
    left_top = approx1[np.argmin(sum)].tolist()
    right_bottom = approx1[np.argmax(sum)].tolist()
    left_bottom = approx1[np.argmax(diff)].tolist()
    right_top = approx1[np.argmin(diff)].tolist()
    return [left_top, right_top, left_bottom, right_bottom] 

    





if __name__ == "__main__":
    mask = Image.open("mask.png").convert("RGB")
    left_top, right_top, left_bottom, right_bottom = mask_to_points(mask)
    print("ok")