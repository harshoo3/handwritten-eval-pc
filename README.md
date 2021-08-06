# HandEval, an automated Handwritten-answer Evaluator

The objective of the project is to approach development in Automated grading/evaluation of  the  Handwritten  Answers  presented  by  a  student. Evaluation takes place using a reference answer provided by the teacher whether in text form or the image of handwritten text. 

The project can be divided into 2 parts namely OCR (Optical Character Recognition) and NLP(Natural Language Processing).

### OCR 

![ocr](./data/images/ocr-pipeline.png)

## NLP

![nlp](./data/images/nlp.png)

## Run demo
 
* Go to the `src` directory 
* Run inference code:
  * Execute `python main.py --student_answer_image PATH_TO_STUDENT_ANSWER_IMAGE --teacher_answer_image PATH_TO_STUDENT_ANSWER_IMAGE` to run the model on those images
  * Execute `python main.py --student_answer_text STRING_CONTAINING_STUDENT_ANSWER --teacher_answer_text STRING_CONTAINING_TEACHER_ANSWER` if text is already decoded to run the model on the text 

## Command line arguments
* `--student_answer_image`: the path to the image with handwritten student answer
* `--teacher_answer_image`: the path to the image with handwritten teacher(reference) answer
* `--student_answer_text`: a string containing student answer
* `--teacher_answer_text`: a string containing teacher(reference) answer
* `--marks`: (Required) an integer denoting maximum marks alloted to the question

The input images and results shown below.

### Input images:

Teacher's Answer           |  Student's Answer
:-------------------------:|:-------------------------:
<img src="./data/images/input1-exp3.png" alt="input2" width="250"/>  |  <img src="./data/images/input1-exp3.png" alt="input2" width="250"/>
<!-- <img src="./data/images/input1-exp3.png" alt="input1" width="170"/>
<img src="./data/images/input1-exp3.png" alt="input2" width="170"/> -->
<!-- ![test](./data/images/input1-exp3.png)

<!-- ![test](./data/images/input2-exp3.png) --> 

### Recognised Text
[recog](./data/images/interpreted-text-exp3.png)

### Keywords found in Teacher's Answer
[keywords](./data/images/keywords-exp3.png)

### Matched Keywords' Location and Marks Evaluation
[eval](./data/images/evaluation-exp3.png)

### Original Images with highlighted matched keywords
[plot](./data/images/matched-keywords-exp3.png)


## Integrate word beam search decoding

The [word beam search decoder](https://repositum.tuwien.ac.at/obvutwoa/download/pdf/2774578) can be used instead of the two decoders shipped with TF.
Words are constrained to those contained in a dictionary, but arbitrary non-word character strings (numbers, punctuation marks) can still be recognized.
The following illustration shows a sample for which word beam search is able to recognize the correct text, while the other decoders fail.

![decoder_comparison](./doc/decoder_comparison.png)

Follow these instructions to integrate word beam search decoding:

1. Clone repository [CTCWordBeamSearch](https://github.com/githubharald/CTCWordBeamSearch)
2. Compile and install by running `pip install .` at the root level of the CTCWordBeamSearch repository
3. Specify the command line option `--decoder wordbeamsearch` when executing `main.py` to actually use the decoder

The dictionary is automatically created in training and validation mode by using all words contained in the IAM dataset (i.e. also including words from validation set) and is saved into the file `data/corpus.txt`.
Further, the manually created list of word-characters can be found in the file `model/wordCharList.txt`.
Beam width is set to 50 to conform with the beam width of vanilla beam search decoding.


## Train model on IAM dataset

### Prepare dataset
Follow these instructions to get the IAM dataset:

* Register for free at this [website](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database)
* Download `words/words.tgz`
* Download `ascii/words.txt`
* Create a directory for the dataset on your disk, and create two subdirectories: `img` and `gt`
* Put `words.txt` into the `gt` directory
* Put the content (directories `a01`, `a02`, ...) of `words.tgz` into the `img` directory

### Run training

* Delete files from `model` directory if you want to train from scratch
* Go to the `src` directory and execute `python main.py --mode train --data_dir path/to/IAM`
* The IAM dataset is split into 95% training data and 5% validation data  
* If the option `--line_mode` is specified, 
  the model is trained on text line images created by combining multiple word images into one  
* Training stops after a fixed number of epochs without improvement

The pretrained word model was trained with this command on a GTX 1050 Ti:
```
python main.py --mode train --fast --data_dir path/to/iam  --batch_size 500 --early_stopping 15
```

And the line model with:
```
python main.py --mode train --fast --data_dir path/to/iam  --batch_size 250 --early_stopping 10
```


### Fast image loading
Loading and decoding the png image files from the disk is the bottleneck even when using only a small GPU.
The database LMDB is used to speed up image loading:
* Go to the `src` directory and run `create_lmdb.py --data_dir path/to/iam` with the IAM data directory specified
* A subfolder `lmdb` is created in the IAM data directory containing the LMDB files
* When training the model, add the command line option `--fast`

The dataset should be located on an SSD drive.
Using the `--fast` option and a GTX 1050 Ti training on single words takes around 3h with a batch size of 500.
Training on text lines takes a bit longer.


## Information about model

The model is a stripped-down version of the HTR system I implemented for [my thesis]((https://repositum.tuwien.ac.at/obvutwhs/download/pdf/2874742)).
What remains is the bare minimum to recognize text with an acceptable accuracy.
It consists of 5 CNN layers, 2 RNN (LSTM) layers and the CTC loss and decoding layer.
For more details see this [Medium article](https://towardsdatascience.com/2326a3487cd5).


## References
* [Build a Handwritten Text Recognition System using TensorFlow](https://towardsdatascience.com/2326a3487cd5)
* [Scheidl - Handwritten Text Recognition in Historical Documents](https://repositum.tuwien.ac.at/obvutwhs/download/pdf/2874742)
* [Scheidl - Word Beam Search: A Connectionist Temporal Classification Decoding Algorithm](https://repositum.tuwien.ac.at/obvutwoa/download/pdf/2774578)

