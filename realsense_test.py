import pyrealsense2 as rs
import numpy as np
import cv2
from edge_contour import color_edge

def realsense_alien_clip(frame, depth_scale, clip_max_dist = 3.5, clip_min_dist = 2.5):
    """
    frame: the frame from realsense
    depth_scale: the depth scale of the camera (data * depth_scale -> meters)
    clip_max_dist: the max distance of the object to the camera
    clip_min_dist: the min distance of the object to the camera
    return: img_dist, img_color(clipped)
    """
    
    # convert meters to distancedata
    clip_max_dist /= depth_scale
    clip_min_dist /= depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)


    aligned_frame = align.process(frame)

    # Get aligned frames
    aligned_depth_frame = aligned_frame.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
    color_frame = aligned_frame.get_color_frame()
    
    depth_image = np.asanyarray(aligned_depth_frame.get_data()) # (480, 640)
    color_image = np.asanyarray(color_frame.get_data()) # (480, 640, 3)
    # Remove background - Set pixels further than clipping_distance to grey
    grey_color = 153
    # depth_image_blur = cv2.GaussianBlur(depth_image, (5, 5), 0)
    depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
    # depth image is 1 channel, color is 3 channels
    # depth_image_3d shape is (480, 640, 3)
    bg_removed = np.where(((depth_image_3d < clip_min_dist) | (depth_image_3d > clip_max_dist)) | (depth_image_3d < 0), grey_color, color_image)
    # the size of bg_removed is (48, 640, 3)
    return bg_removed, depth_image, aligned_depth_frame


if __name__ == "__main__":
    # Create a pipeline

    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: ", depth_scale)

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    # clipping_distance_in_meters = 3  # 1 meter
    # clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Streaming loop
    try:
        while True:
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()
            # frames.get_depth_frame() is a 640x360 depth image

            bg_removed, depth_image = realsense_alien_clip(frames, depth_scale, clip_max_dist = float('inf'), clip_min_dist = 0)
            
            # fake color of depth
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            
            # 数据放入yolo得到裁切后的值
            yolo_result, pos_list = MyDetect(bg_removed)


            if yolo_result is not None:
                # 左-绿
                cv2.line(bg_removed, (pos_list[0], pos_list[2]), (pos_list[0], pos_list[3]), (0, 255, 0), 10)
                # 右-红
                cv2.line(bg_removed, (pos_list[1], pos_list[2]), (pos_list[1], pos_list[3]), (0, 0, 255), 10)
                # 上-蓝
                cv2.line(bg_removed, (pos_list[0], pos_list[2]), (pos_list[1], pos_list[2]), (255, 0, 0), 10)
                # 下-白
                cv2.line(bg_removed, (pos_list[0], pos_list[3]), (pos_list[1], pos_list[3]), (255, 255, 255), 10)
                
                # 根据裁切后的值进行角点检测，返回四边形的四个角点的相对位置（相对裁切图片而言）[height, width]
                corner_pos_list = color_edge(yolo_result) # left_top, right_top, left_bottom, right_bottom
                # 对得到的边界进行坐标转换
                # 集合起来坐标变换
                if corner_pos_list is None:
                    continue
                for corner_pos in corner_pos_list:
                    corner_pos[0] += pos_list[2]
                    corner_pos[1] += pos_list[0]
                print(corner_pos_list)
                # 画出角点
                for corner_pos in corner_pos_list:
                    cv2.circle(bg_removed, (corner_pos[1], corner_pos[0]), 5, (0, 0, 255), -1)
                
            # combine clipped_image and depth
            images = np.hstack(( bg_removed, depth_colormap))
            
            cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
            cv2.imshow('Align Example', images)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()
