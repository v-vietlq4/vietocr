from re import I
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import ImageDraw, Image
import argparse
random.seed(0)

class_id_to_name_mapping = {
    0: 'top_left',
    1: 'top_right',
    2: 'bottom_left',
    3: 'bottom_right'
}


def plot_bounding_box(image, annotation_list):
    annotations = np.array(annotation_list)
    w, h = image.size
    plotted_image = ImageDraw.Draw(image)

    transformed_annotations = np.copy(annotations)
    transformed_annotations[:, [1, 3]] = annotations[:, [1, 3]]*w
    transformed_annotations[:, [2, 4]] = annotations[:, [2, 4]]*h

    transformed_annotations[:, 1] = transformed_annotations[:,
                                                            1] - (transformed_annotations[:, 3] / 2)
    transformed_annotations[:, 2] = transformed_annotations[:,
                                                            2] - (transformed_annotations[:, 4] / 2)
    transformed_annotations[:, 3] = transformed_annotations[:,
                                                            1] + transformed_annotations[:, 3]
    transformed_annotations[:, 4] = transformed_annotations[:,
                                                            2] + transformed_annotations[:, 4]

    for ann in transformed_annotations:
        obj_cls, x0, y0, x1, y1 = ann
        plotted_image.rectangle(((x0, y0), (x1, y1)), fill='#ffff33')

        plotted_image.text(
            (x0, y0 - 10), class_id_to_name_mapping[(int(obj_cls))], fill='red')

    plt.imshow(np.array(image))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotationFile', type=str,
                        default='/home/vietlq4/PaddleOCR/dataset/train/yolo/facare_VID_20221014_153432_iframe180.txt', help='The annotation file of YOLO format')
    args = parser.parse_args()

    with open(args.annotationFile) as f:
        annotation_list = f.read().split('\n')[:-1]
        annotation_list = [x.split(" ") for x in annotation_list]
        annotation_list = [[float(y) for y in x] for x in annotation_list]

    image_file = '/home/vietlq4/PaddleOCR/dataset/train/android/facare/VID_20221014_153432/facare_VID_20221014_153432_iframe180.jpg'
    image = Image.open(image_file)

    plot_bounding_box(image, annotation_list)
