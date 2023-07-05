import cv2
import numpy as np
# # 读取图像
# def get_edge_contour(img):
#     threshold1 = 100
#     threshold2 = 200
#     len_threshold = 2
#     edges = cv2.Canny(img, threshold1, threshold2)

#     cv2.imwrite('edges.png', edges)

#     # 查找轮廓
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    
#     # 显示所有轮廓
#     mask = np.zeros(img.shape)
#     for c in contours:
#         # 过滤小面积
#         if (cv2.contourArea(c) < len_threshold ** 2):
#             continue
    
#         cv2.drawContours(img, [c], 0, (0, 255, 0), 1)
    
#     cv2.imwrite('contours.png', img)


def color_edge(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    height, width, _ = img.shape
    scale = (height + width)/2
    
    # cv2.imshow("hsv", hsv)
    minBlue = np.array([100, 15, 46])
    maxBlue = np.array([124, 255, 255])
    
    # 确定蓝色区域
    mask = cv2.inRange(hsv, minBlue, maxBlue)
    # cv2.imwrite("mask.png", mask)
    
    # 通过按位与获取蓝色区域
    blue_img = cv2.bitwise_and(img, img, mask=mask)
    cv2.imwrite("blue.png", blue_img)

    # 将mask进行形态学处理消除内部外部的噪点
    kernel_size1 = int(scale / 60)
    kernel_size2 = int(scale / 16)
    if kernel_size1 == 0 or kernel_size2 == 0:
        return None
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_size1,kernel_size1))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_size2,kernel_size2))
    #定义矩形结构元素
    # erode1 = cv2.erode(mask,kernel,iterations=1)
    # cv2.imwrite("erode1.png", erode1)
    open0 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,iterations=2)
    cv2.imwrite("open0.png", open0)
    
    closed1 = cv2.morphologyEx(open0, cv2.MORPH_CLOSE, kernel2,iterations=2)
    cv2.imwrite("closed1.png", closed1)

    open1 = cv2.morphologyEx(closed1, cv2.MORPH_OPEN, kernel2,iterations=2)
    cv2.imwrite("open1.png", open1)

    # 提取边界
    ret, binary = cv2.threshold(open1,127,255,cv2.THRESH_BINARY)

    binary = np.float32(binary)
    dst = cv2.cornerHarris(binary,4,5,0.04)
    dst = cv2.dilate(dst,None)
    # dst: height * width
    
    candidate_pos = np.array(np.where(dst > 0.2 * dst.max())).transpose()
    if candidate_pos.shape[0] == 0:
        return None
    # pos[0]->height pos[1]->width
    left_top = candidate_pos[0]
    right_top = candidate_pos[0]
    left_bottom = candidate_pos[0]
    right_bottom = candidate_pos[0]
    # 遍历每个点，找到距离四角最近的点
    # 计算四个角点的坐标
    pos_plus = candidate_pos[:, 0] + candidate_pos[:, 1]
    pos_minus = candidate_pos[:, 0] - candidate_pos[:, 1]
    left_top_arg = np.argmin(pos_plus)
    right_bottom_arg = np.argmax(pos_plus)
    left_bottom_arg = np.argmax(pos_minus)
    right_top_arg = np.argmin(pos_minus)

    left_top = candidate_pos[left_top_arg]
    right_bottom = candidate_pos[right_bottom_arg]
    left_bottom = candidate_pos[left_bottom_arg]
    right_top = candidate_pos[right_top_arg]
    
    return [left_top, right_top, left_bottom, right_bottom] 




if __name__ == "__main__":
    # img = cv2.imread('0.png', cv2.THRESH_BINARY)
    # cv2.imwrite('1.png', img)
    # get_edge_contour(img)
    img = cv2.imread('0.png')
    color_edge(img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


