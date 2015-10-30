## Running demo.py using dlib's selective search:

I changed some code in demo.py to use dlib’s selective search to generate proposal bounding boxes, so no *_boxes.mat is needed for the demo or other pictures.

To run the changed demo, you need to compile dlib for python:

1. Download [dlib](http://dlib.net/)
2. if you are on ubuntu:

        sudo apt-get install libboost-python-dev cmake
        cd dlib-18.15/python_examples
        ./compile_dlib_python_module.bat 
        sudo cp dlib.so /usr/local/lib/python2.7/dist-packages/
   If import dlib in python shell gives no error, the dlib shoud be ready.
   
3. Then just install fast-rcnn and run demo.py. A dog should be detected in the shown image. Change the image to your own to run detection by fast-rcnn(only the 20 classes in Pascal Voc).

## Training and testing on INRIA without MATLAB
*All codes training on INRIA are forked from the reference page, I only change some code to get rid of the MATLAB*

1. Using generate_bbox.py to generate train.mat and copy it to the dataset path as the reference page says.
2. Train the net on INRIA as the reference page.
3. Change all 'train.mat' in generate_bbox.py to 'test.mat'
4. Using generate_bbox.py to generate test.mat and copy it to the dataset path as the reference page says.
5. Running evaluation.py to caculate accuracy on the test dataset.

*All paths in the files are sepecific to my own computer, sorry for my laziness to change them. You might need to find them and change to your own. I will focus on training fast rcnn on ImageNet, when that is done, I might get rid of the sepecific paths and make this project more general*

## Reference:
https://github.com/zeyuanxy/fast-rcnn/tree/master/help/train
