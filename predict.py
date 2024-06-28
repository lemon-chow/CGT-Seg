import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from utils.data_loading import BasicDataset
from unet import UNet, SEUNet, UResnet34, UResnet50, UResnet101, UResnet152
from utils.utils import plot_img_and_mask

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            output = F.softmax(output, dim=1)
            # remove the first channel (background)
            # output = output[:, 1:, :, :]
            # remove the second channel (vessels)
            # output = output[:, :1, :, :]
            # output = output[:, :1, :, :]
            print(output.shape)
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='checkpoints/checkpoint_epoch24.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    
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


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = args.input
    out_files = get_output_filenames(args)

    # net = SEUNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    # net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    net = UResnet34(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)
        true_mask = Image.open(filename.replace('image_1', 'mask'))

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask, mask_values)
            # map labels in result to colors
            result.putpalette([
                0, 0, 0,  # Black background
                255, 255, 255,  # Class 1
                0, 0, 255,  # Class 2
                0, 255, 0,  # Class 3
            ])
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

            # compare this to the original image and mask
            
            plt.figure(figsize=(10, 10))
            plt.subplot(1, 3, 1)
            plt.imshow(img)
            plt.axis('off')
            plt.title('Original Image')
            plt.subplot(1, 3, 2)
            plt.imshow(result)
            plt.axis('off')
            plt.title('Predicted Mask')
            plt.subplot(1, 3, 3)
            
            true_mask = mask_to_image(np.asarray(true_mask), mask_values)
            true_mask.putpalette([
                0, 0, 0,  # Black background
                255, 255, 255,  # Class 1
                0, 0, 255,  # Class 2
                0, 255, 0,  # Class 3
            ])
            plt.imshow(true_mask)
            plt.axis('off')
            plt.title('True Mask')
            plt.show()
            plt.savefig('res_comparsion')


        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)