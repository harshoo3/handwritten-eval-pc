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

from dataloader_iam import DataLoaderIAM, Batch
from model import Model, DecoderType
from preprocessor import Preprocessor
from deslant import deslant
from deslant_main import parse_args
from nlp_main import nlp_main

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


def write_summary(char_error_rates: List[float], word_accuracies: List[float]) -> None:
    """Writes training summary file for NN."""
    with open(FilePaths.fn_summary, 'w') as f:
        json.dump({'charErrorRates': char_error_rates, 'wordAccuracies': word_accuracies}, f)


def train(model: Model,
          loader: DataLoaderIAM,
          line_mode: bool,
          early_stopping: int = 25) -> None:
    """Trains NN."""
    epoch = 0  # number of training epochs since start
    summary_char_error_rates = []
    summary_word_accuracies = []
    preprocessor = Preprocessor(get_img_size(line_mode), data_augmentation=True, line_mode=line_mode)
    best_char_error_rate = float('inf')  # best valdiation character error rate
    no_improvement_since = 0  # number of epochs no improvement of character error rate occurred
    # stop training after this number of epochs without improvement
    while True:
        epoch += 1
        print('Epoch:', epoch)

        # train
        print('Train NN')
        loader.train_set()
        while loader.has_next():
            iter_info = loader.get_iterator_info()
            batch = loader.get_next()
            batch = preprocessor.process_batch(batch)
            loss = model.train_batch(batch)
            print(f'Epoch: {epoch} Batch: {iter_info[0]}/{iter_info[1]} Loss: {loss}')

        # validate
        char_error_rate, word_accuracy = validate(model, loader, line_mode)

        # write summary
        summary_char_error_rates.append(char_error_rate)
        summary_word_accuracies.append(word_accuracy)
        write_summary(summary_char_error_rates, summary_word_accuracies)

        # if best validation accuracy so far, save model parameters
        if char_error_rate < best_char_error_rate:
            print('Character error rate improved, save model')
            best_char_error_rate = char_error_rate
            no_improvement_since = 0
            model.save()
        else:
            print(f'Character error rate not improved, best so far: {char_error_rate * 100.0}%')
            no_improvement_since += 1

        # stop training if no more improvement in the last x epochs
        if no_improvement_since >= early_stopping:
            print(f'No more improvement since {early_stopping} epochs. Training stopped.')
            break


def validate(model: Model, loader: DataLoaderIAM, line_mode: bool) -> Tuple[float, float]:
    """Validates NN."""
    print('Validate NN')
    loader.validation_set()
    preprocessor = Preprocessor(get_img_size(line_mode), line_mode=line_mode)
    num_char_err = 0
    num_char_total = 0
    num_word_ok = 0
    num_word_total = 0
    while loader.has_next():
        iter_info = loader.get_iterator_info()
        print(f'Batch: {iter_info[0]} / {iter_info[1]}')
        batch = loader.get_next()
        batch = preprocessor.process_batch(batch)
        recognized, _ = model.infer_batch(batch)

        print('Ground truth -> Recognized')
        for i in range(len(recognized)):
            num_word_ok += 1 if batch.gt_texts[i] == recognized[i] else 0
            num_word_total += 1
            dist = editdistance.eval(recognized[i], batch.gt_texts[i])
            num_char_err += dist
            num_char_total += len(batch.gt_texts[i])
            print('[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' + batch.gt_texts[i] + '"', '->',
                  '"' + recognized[i] + '"')

    # print validation result
    char_error_rate = num_char_err / num_char_total
    word_accuracy = num_word_ok / num_word_total
    print(f'Character error rate: {char_error_rate * 100.0}%. Word accuracy: {word_accuracy * 100.0}%.')
    return char_error_rate, word_accuracy


def infer(model: Model, fn_img: Path,new_new_list:List, list_of_recog_answer: List) -> None:
    # """Recognizes text in image provided by file path."""
    # """Recognizes text in image provided by file path."""
    # img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
    # assert img is not None

    # preprocessor = Preprocessor(get_img_size(), dynamic_width=True, padding=16)
    # img = preprocessor.process_img(img)

    # batch = Batch([img], None, 1)
    # recognized, probability = model.infer_batch(batch, True)
    # print(f'Recognized: "{recognized[0]}"')
    # # print(f'Probability: {probability[0]}')
    # keras_file = "My_saved_Model.h5"
    # keras.models.save_model(model , keras_file)
    # tf.keras.models.save_model('../my_model',save_format='h5')
    img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
    img = img[820:2800,200:2800]
    # img = cv2.resize(img, (600, 400))
    print(img.shape)
    copy = img.copy()
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
    # new_contours = [contours[i] for i in cntr_index_LtoR] #make new contours based on sorted list
    # line_seg=[[]] #making a list of contours that segregates lines
    # curr_y= cv2.boundingRect(new_contours[0])[1] 
    # index = 0
    # for c in new_contours:
    #     x, y, w, h = cv2.boundingRect(c) 
    #     if(y> curr_y+25): #distinguishing a line based on a threshold y co-ordinate
    #         curr_y=y
    #         index+=1
    #         line_seg.append([])
    #     line_seg[index].append(c)

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

    max_y_h=[]
    max_y_h.append(0)
    counter2 =1
    for i in line_seg:
        # print('hi'+str(counter2))

        max_y_h.append(0)
        for c in i:
            # print('hi'+str(counter2))
            x, y, w, h = cv2.boundingRect(c)
            # print(y+h)
            if(y+h>max_y_h[counter2]):
                max_y_h[counter2]=y+h
        counter2+=1

    # print('hi1')
    for i in range(1,len(max_y_h)):
        # print('hi'+str(i))
        print(max_y_h[i])
        # print(max_y_h[i-1])
        # print(copy2.shape)
        line_img = copy[max_y_h[i-1]:max_y_h[i],:]
        # cv2.imshow('line_img',line_img)


    # print(new_list[0])
    ROI_number=0
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
            cv2.imwrite('D:\summer-project\model3\SimpleHTR\data\slanted\img_{}.png'.format(ROI_number),ROI)
            ROI = deslant(ROI,
                      optim_algo=parsed.optim_algo,
                      lower_bound=parsed.lower_bound,
                      upper_bound=parsed.upper_bound,
                      num_steps=parsed.num_steps,
                      bg_color=parsed.bg_color)
            # cv2.imshow('ROI',ROI)
            cv2.imwrite('D:\summer-project\model3\SimpleHTR\data\deslanted\img_{}.png'.format(ROI_number),ROI)
            ROI_number += 1
            assert ROI is not None

            preprocessor = Preprocessor(get_img_size(), dynamic_width=True, padding=16)
            ROI = preprocessor.process_img(ROI)

            batch = Batch([ROI], None, 1)
            recognized, probability = model.infer_batch(batch, True)
            answer+=str(recognized[0])+' '
            list_of_recog_answer.append(str(recognized[0]))

        # print(f'Recognized: "{recognized[0]}"')
        # print(f'Probability: {probability[0]}')

    print(answer)
    return answer

def main():
    """Main function."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', choices=['train', 'validate', 'infer'], default='infer')
    parser.add_argument('--decoder', choices=['bestpath', 'beamsearch', 'wordbeamsearch'], default='bestpath')
    parser.add_argument('--batch_size', help='Batch size.', type=int, default=100)
    parser.add_argument('--data_dir', help='Directory containing IAM dataset.', type=Path, required=False)
    parser.add_argument('--fast', help='Load samples from LMDB.', action='store_true')
    parser.add_argument('--line_mode', help='Train to read text lines instead of single words.', action='store_true')
    parser.add_argument('--img_file', help='Image used for inference.', type=Path, default='../data/word.png')
    parser.add_argument('--student_answer', help='Student Answer Image used for inference.', type=Path, default='../data/try4.jpeg')
    parser.add_argument('--teacher_answer', help='Teacher Answer Image used for inference.', type=Path, default='../data/line.png')
    parser.add_argument('--img_files', help='Images used for inference.', type=Tuple, default=('../data/try4.jpeg','../data/line.png'))
    parser.add_argument('--early_stopping', help='Early stopping epochs.', type=int, default=25)
    parser.add_argument('--dump', help='Dump output of NN to CSV file(s).', action='store_true')
    args = parser.parse_args()

    # set chosen CTC decoder
    decoder_mapping = {'bestpath': DecoderType.BestPath,
                       'beamsearch': DecoderType.BeamSearch,
                       'wordbeamsearch': DecoderType.WordBeamSearch}
    decoder_type = decoder_mapping[args.decoder]

    # meta_path = '../model/snapshot-33.meta' # Your .meta file
    # output_node_names = [n.name for n in tf.compat.v1.get_default_graph().as_graph_def().node]    # Output nodes

    # with tf.compat.v1.Session() as sess:
    #     # Restore the graph
    #     saver = tf.compat.v1.train.Saver(max_to_keep=1)
    #     # saver = tf.compat.v1.train.import_meta_graph(meta_path)

    #     # Load weights
    #     saver.restore(sess,tf.train.latest_checkpoint('../model/'))

        # Freeze the graph
        # frozen_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
        #     sess,
        #     sess.graph_def,
        #     output_node_names)

        # # Save the frozen graph
        # with open('output_graph.pb', 'wb') as f:
        #     f.write(frozen_graph_def.SerializeToString())

    # print('done')
    # sess = tf.compat.v1.Session()  # TF session

    # model_dir = '../model/'
    # sess.run(tf.compat.v1.global_variables_initializer())
    # latest_snapshot = tf.train.latest_checkpoint(model_dir)
    # saver = tf.compat.v1.train.Saver()  # saver saves model to file
    # if latest_snapshot:
    #     print('Init with stored values from ' + latest_snapshot)
    #     saver.restore(sess, latest_snapshot)
    # print('done')
    # train or validate on IAM dataset
    if args.mode in ['train', 'validate']:
        # load training data, create TF model
        loader = DataLoaderIAM(args.data_dir, args.batch_size, fast=args.fast)
        char_list = loader.char_list

        # when in line mode, take care to have a whitespace in the char list
        if args.line_mode and ' ' not in char_list:
            char_list = [' '] + char_list

        # save characters of model for inference mode
        open(FilePaths.fn_char_list, 'w').write(''.join(char_list))

        # save words contained in dataset into file
        open(FilePaths.fn_corpus, 'w').write(' '.join(loader.train_words + loader.validation_words))

        # execute training or validation
        if args.mode == 'train':
            model = Model(char_list, decoder_type)
            train(model, loader, line_mode=args.line_mode, early_stopping=args.early_stopping)
        elif args.mode == 'validate':
            model = Model(char_list, decoder_type, must_restore=True)
            validate(model, loader, args.line_mode)

    # infer text on test image
    elif args.mode == 'infer':
        model = Model(list(open(FilePaths.fn_char_list).read()), decoder_type, must_restore=True, dump=args.dump)
        # print(args.img_files[0])
        # infer(model, args.img_file)
        list_of_teacher_answer=[]
        student_answer_locations=[]
        teacher_answer_locations=[]
        list_of_student_answer=[]
        start_time = time.time()
        # python main.py --student_answer D:\summer-project\handwritten-dataset\formsA-D.tgz\formsA-D\a01-096u.png --teacher_answer D:\summer-project\handwritten-dataset\formsA-D.tgz\formsA-D\a01-026.png
        teacher_answer=infer(model, args.teacher_answer,teacher_answer_locations,list_of_teacher_answer)
        student_answer=infer(model, args.student_answer,student_answer_locations,list_of_student_answer)
        # print(student_answer)
        # print(teacher_answer)
        # print(list_of_student_answer)
        # print(student_answer_locations)
        nlp_main(teacher_answer,student_answer)
        print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    main()
