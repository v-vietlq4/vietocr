from ast import Try, arg
from cgi import test
from email.mime import image
from email.policy import default
import os
from re import I
import numpy as np
import glob
from shutil import copyfile
import argparse
import re
import shutil
import random


def getImagesInDir(dirPath):
    imageList = []
    for root, dirs, files in os.walk(dirPath):
        for file in files:
            if file.endswith('.jpg'):
                imageList.append(os.path.join(root, file))
    return imageList


def splitTrainValTest(source, dest, args):
    trainDir = os.path.join(dest, 'train')
    testDir = os.path.join(dest, 'test')
    valDir = os.path.join(dest, 'val')

    os.makedirs(trainDir, exist_ok=True)
    os.makedirs(valDir, exist_ok=True)
    os.makedirs(testDir, exist_ok=True)

    images = getImagesInDir(source)
    numImages = len(images)
    trainValTestRatio = args.trainValTestRatio.split(":")
    trainRatio = eval(trainValTestRatio[0]) / 10
    valRatio = trainRatio + eval(trainValTestRatio[1]) / 10
    count = 0
    while(len(images) != 0):
        count = count + 1
        idx = random.randint(0, len(images) - 1)
        curRatio = count/numImages
        filePath = images[idx].replace(str(source), '')
        xmlPath = filePath.replace('.jpg', '.xml')

        if curRatio < trainRatio:
            try:
                os.makedirs(os.path.dirname(os.path.join(
                    trainDir, filePath)), exist_ok=True)
                os.makedirs(os.path.dirname(os.path.join(
                    trainDir, xmlPath)), exist_ok=True)
                shutil.copy(os.path.join(source, filePath),
                            os.path.join(trainDir, filePath))
                shutil.copy(os.path.join(source, xmlPath),
                            os.path.join(trainDir, xmlPath))
            except:
                print("no label")
        elif curRatio >= trainRatio and curRatio < valRatio:
            try:
                os.makedirs(os.path.dirname(
                    os.path.join(valDir, filePath)), exist_ok=True)
                os.makedirs(os.path.dirname(
                    os.path.join(valDir, xmlPath)), exist_ok=True)
                shutil.copy(os.path.join(source, filePath),
                            os.path.join(valDir, filePath))
                shutil.copy(os.path.join(source, xmlPath),
                            os.path.join(valDir, xmlPath))
            except:
                "no label"
        else:
            try:
                os.makedirs(os.path.dirname(os.path.join(
                    testDir, filePath)), exist_ok=True)
                os.makedirs(os.path.dirname(
                    os.path.join(testDir, xmlPath)), exist_ok=True)
                shutil.copy(os.path.join(source, filePath),
                            os.path.join(testDir, filePath))
                shutil.copy(os.path.join(source, xmlPath),
                            os.path.join(testDir, xmlPath))
            except:
                "no label"
        images.remove(images[idx])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainValTestRatio', type=str,
                        default="6:2:2", help="ratio of trainset:valset:testset")
    parser.add_argument('--datasetRootPath', type=str,
                        default="./data/", help="path of root dataset")
    parser.add_argument('--destRoot', type=str,
                        default='./dataset/', help="Dest of train val test root")

    args = parser.parse_args()

    imageList = getImagesInDir(args.datasetRootPath)
    print(len(imageList))
    os.makedirs(args.destRoot, exist_ok=True)

    splitTrainValTest(args.datasetRootPath, args.destRoot, args)

    print(imageList[0].replace('.jpg', '.xml'))
