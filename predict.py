import argparse
import logging
import os
import PIL
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import time

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask
import cProfile, pstats, io

def predict_img(net,
                full_img,
                device,
                W = 96,
                H = 96,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, W , H , is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='checkpoints/checkpoint_epoch5.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', default='data/image', metavar='data/image', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='result', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
    
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)

def Predict_Mask(model_path, imput_img, W:int = 320, H:int = 320, threshold:float = 0.05):
        
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(n_channels=3, n_classes=1, bilinear=False)    
    net.to(device=device)
    state_dict = torch.load(model_path, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)
    
    # Predict
    mask = predict_img(net=net,
                        full_img=imput_img,
                        W = W,
                        H = H,
                        out_threshold=threshold,
                        device=device)
    result = mask_to_image(mask, mask_values)
    return result    

if __name__ == "__main__":
    model_path = 'model15.pth'
    img = Image.open('./data/image/1.jpg')
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(n_channels=3, n_classes=1, bilinear=False)    
    net.to(device=device)
    state_dict = torch.load(model_path, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)
    
    # cProfile
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Predict
    for i in range(100):
        mask = predict_img(net=net,
                            full_img=img,
                            W = 640,
                            H = 640,
                            out_threshold=0.005,
                            device=device)
        result = mask_to_image(mask, mask_values)
    
    profiler.disable()
    # 'tottime' 运行总时间，'cumtime' 运行总时间+调用函数运行时间
    # 'stdname', 'calls', 'time', 'cumulative' 
    sortby = 'cumulative'
    ps = pstats.Stats(profiler).sort_stats(sortby)
    with open('Predict_time.txt', 'w') as f:
        ps.print_stats()
        ps.stream = f
        ps.print_stats()
    

# if __name__ == '__main__':
#     logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
#     in_files = ['./data/image/1.jpg', './data/image/2.jpg', './data/image/3.jpg']
#     out_files = ['./result/1_OUT.png', './result/2_OUT.png', './result/3_OUT.png']
#     net = UNet(n_channels=3, n_classes=1, bilinear=False)

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logging.info(f'Loading model :checkpoints/model7.pth')
#     logging.info(f'Using device {device}')

#     net.to(device=device)
#     state_dict = torch.load('./checkpoints/model7.pth', map_location=device)
#     mask_values = state_dict.pop('mask_values', [0, 1])
#     net.load_state_dict(state_dict)

#     logging.info('Model loaded!')

#     for i, filename in enumerate(in_files):
#         logging.info(f'Predicting image {filename} ...')
#         img = Image.open(filename)

#         mask = predict_img(net=net,
#                            full_img=img,
#                            W = 960,
#                            H = 960,
#                            out_threshold=0.05,
#                            device=device)

#         if True:
#             out_filename = out_files[i]
#             result = mask_to_image(mask, mask_values)
            
#             print(result.size)
#             result.save(out_filename)
#             logging.info(f'Mask saved to {out_filename}')

#         if True:
#             logging.info(f'Visualizing results for image {filename}, close to continue...')
#             plot_img_and_mask(img, mask)
