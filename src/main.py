import argparse
import json
from typing import Tuple, List
import numpy as np
import cv2
import editdistance
import keras
from path import Path
import tensorflow as tf
import time
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

from dataloader_iam import DataLoaderIAM, Batch
from model import Model, DecoderType
from preprocessor import Preprocessor
from deslant import deslant
from deslant_main import parse_args
from nlp_main import nlp_main
from nlp_main import Regex
class FilePaths:
    """Filenames and paths to data."""
    fn_char_list = '../model/charList.txt'
    fn_summary = '../model/summary.json'
    fn_corpus = '../data/corpus.txt'

def get_img_height() -> int:
    """Fixed height for NN."""
    return 32


def get_img_size(line_mode: bool = False) -> Tuple[int, int]:
    """Height is fixed for NN, width is set according to training mode (single words or text lines)."""
    if line_mode:
        return 256, get_img_height()
    return 128, get_img_height()

def infer(model: Model, fn_img: Path,locations:dict) -> None:
    # """Recognizes text in image provided by file path."""

    img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
    img = img[750:2800,200:2800]
    # img = cv2.resize(img, (600, 400))
    # print(img.shape)
    copy = img.copy()
    copy3 = cv2.imread(fn_img)
    copy3=copy3[750:2800,200:2800]
    parsed = parse_args()
    answer=''

    cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU,img)  #to convert image into grayscale
    custom_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 20)) #custom kernel which governs the word segregation
    threshed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, custom_kernel) # transforms the image according to the kernel

    contours, hier = cv2.findContours(threshed, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) # find contours in the image
    # and return only the external contours
    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        # draw a white rectangle to visualize the bounding rect
        cv2.rectangle(img, (x, y), (x + w, y + h), 255, 1)  

    # cv2.imshow('img',img)
    cv2.drawContours(img, contours, -1, (255, 255, 0), 1)

    cntr_index_LtoR = np.argsort([cv2.boundingRect(i)[1] for i in contours])  #sorting the 
    # indices of the contours based on y coordinate of contours
    new_contours = [contours[i] for i in cntr_index_LtoR] #make new contours based on sorted list
    line_seg=[[]] #making a list of contours that segregates lines
    curr_y= cv2.boundingRect(new_contours[0])[1] 
    avg_h = cv2.boundingRect(new_contours[0])[1]
    index = 0
    counter = 0
    for c in new_contours:
        x, y, w, h = cv2.boundingRect(c)
        # print(y)
        counter+=1
        avg_h=(avg_h*(counter-1)+h)/counter
    
    counter = 0
    for c in new_contours:
        x, y, w, h = cv2.boundingRect(c)
        # print(y)
        if(y> curr_y+avg_h+20): #distinguishing a line based on a threshold y co-ordinate
            # print(counter)
            # print(y)
            curr_y=y
            index+=1
            line_seg.append([])
        counter+=1
        line_seg[index].append(c)

    for j in range(index+1):
        args = np.argsort([cv2.boundingRect(i)[0] for i in line_seg[j]])
        # print(abcd)
        word_seg =[line_seg[j][i] for i in args]
        # for c in contours:
        for c in word_seg:
            x,y,w,h = cv2.boundingRect(c)
            # print(y)
            ROI = copy[y:y+h, x:x+w]
            # cv2.imshow('ROI',ROI)
            # cv2.imwrite('D:\summer-project\model3\SimpleHTR\data\slanted\img_{}.png'.format(ROI_number),ROI)
            ROI = deslant(ROI,
                      optim_algo=parsed.optim_algo,
                      lower_bound=parsed.lower_bound,
                      upper_bound=parsed.upper_bound,
                      num_steps=parsed.num_steps,
                      bg_color=parsed.bg_color)
            # cv2.imshow('ROI',ROI)
            # cv2.imwrite('D:\summer-project\model3\SimpleHTR\data\deslanted\img_{}.png'.format(ROI_number),ROI)
            # ROI_number += 1
            assert ROI is not None

            preprocessor = Preprocessor(get_img_size(), dynamic_width=True, padding=16)
            ROI = preprocessor.process_img(ROI)

            batch = Batch([ROI], None, 1)
            recognized, probability = model.infer_batch(batch, True)
            answer+=str(recognized[0])+' '
            # list_of_recog_answer.append(str(recognized[0]))
            temp_string =Regex(str(recognized[0])).strip()
            # print(temp_string)
            locations.setdefault(temp_string, [])
            locations[temp_string].append(c)

        # print(f'Recognized: "{recognized[0]}"')
        # print(f'Probability: {probability[0]}')

    # print(answer)
    return answer,copy3

def save_matched_contours(keyword_match_list:list,locations:dict,image,name:str,colors_tuple:list):
    index=0
    for words in keyword_match_list:
        try:
            for c in locations[words]:
                x,y,w,h = cv2.boundingRect(c)
                cv2.rectangle(image, (x, y), (x + w, y + h),colors_tuple[index], 3)
        except:
            pass
        index+=1
    
    # cv2.imshow(name,image)
    cv2.imwrite('D:\summer-project\HandEval\data\matched-'+name+'.png',image)
    print(image.shape)
    return image

def plot_images(img1,img2):
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # fig.suptitle('Horizontally stacked subplots')
    # ax1.plot(img1)
    # ax2.plot(img2)
    fig = plt.figure()
    ax1=fig.add_subplot(1, 2, 1)
    ax1.title.set_text("Teacher's answer")
    plt.imshow(img1)
    ax2=fig.add_subplot(1, 2, 2)
    ax2.title.set_text("Student's answer")
    plt.imshow(img2)
    plt.title
    plt.show()

def plot_an_image(img):
    plt.imshow(img)
    plt.show()

def create_colors(s:int):
    colors_list = [list(np.random.choice(range(256), size=3))for k in range(s)]
    colors_tuple=[(colors[0].item(),colors[1].item(),colors[2].item()) for colors in colors_list]

    return colors_tuple

def main():
    """Main function."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--marks', help='Marks alloted for this question', type=int, required=True)
    parser.add_argument('--student_answer_text', help='Student Answer String used for inference.', type=str, default=' ')
    parser.add_argument('--teacher_answer_text', help='Teacher Answer String used for inference.', type=str, default=' ')
    parser.add_argument('--student_answer_image', help='Student Answer Image used for inference.', type=Path, default='../data/try4.jpeg')
    parser.add_argument('--teacher_answer_image', help='Teacher Answer Image used for inference.', type=Path, default='../data/line.png')
    args = parser.parse_args()

    decoder_type = DecoderType.WordBeamSearch
    marks = args.marks

    model = Model(list(open(FilePaths.fn_char_list).read()), decoder_type, must_restore=True, dump=args.dump)
    student_answer_locations={}
    teacher_answer_locations={}
    start_time = time.time()
    student_image_present = True
    teacher_image_present = True
    # python main.py --student_answer_image D:\summer-project\handwritten-dataset\formsA-D.tgz\formsA-D\a01-096u.png --teacher_answer_image D:\summer-project\handwritten-dataset\formsA-D.tgz\formsA-D\a01-026.png --marks 5
    if args.teacher_answer_text == ' ':
        teacher_answer,teacher_img=infer(model, args.teacher_answer_image,teacher_answer_locations)
    else:
        teacher_answer = args.teacher_answer_text
        teacher_image_present=False


    if args.student_answer_text == ' ':
        student_answer,student_img=infer(model, args.student_answer_image,student_answer_locations)
    else:
        student_answer = args.student_answer_text
        student_image_present=False
    
    # print(student_answer)
    # print(teacher_answer)
    # print(teacher_answer_locations)
    # print(student_answer_locations)
    keyword_match_list=nlp_main(teacher_answer,student_answer,marks)
    # print(keyword_match_list)

    colors_tuple = create_colors(len(keyword_match_list))
    if teacher_image_present:
        img1=save_matched_contours(keyword_match_list,teacher_answer_locations,teacher_img,'teacher',colors_tuple)
        if not student_image_present:
            plot_an_image(img1)
    if student_image_present:
        img2=save_matched_contours(keyword_match_list,student_answer_locations,student_img,'student',colors_tuple)
        if not teacher_image_present:
            plot_an_image(img2)

    if student_image_present and teacher_image_present:
        plot_images(img1,img2)

    
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    main()
