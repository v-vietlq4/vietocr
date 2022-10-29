import sys
import argparse
import numpy as np
import cv2
import os
import subprocess


def getFrameTypes(video_fn):
    command = 'ffprobe -v error -show_entries frame=pict_type -of default=noprint_wrappers=1'.split()
    out = subprocess.check_output(command + [video_fn]).decode()
    frame_types = out.replace('pict_type=', '').split()
    return zip(range(len(frame_types)), frame_types)


def extractImages(pathIn, pathOut):
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    success, image = vidcap.read()
    success = True
    frames_types = getFrameTypes(pathIn)
    iframes = [x[0] for x in frames_types if x[1] == 'I']
    os.makedirs(pathOut, exist_ok=True)
    if iframes:
        basename = os.path.splitext(os.path.basename(pathIn))[0]
        cap = cv2.VideoCapture(pathIn)

        for frame_no in iframes:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            outname = pathOut + "/omron_" + basename + \
                '_iframe' + str(frame_no) + '.jpg'
            cv2.imwrite(outname, frame)

        cap.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pathIn", help="path to video", type=str)
    parser.add_argument("--pathOut", help="path to image", type=str)
    args = parser.parse_args()
    extractImages(args.pathIn, args.pathOut)
