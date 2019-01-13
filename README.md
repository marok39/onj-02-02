# Assignment 2: IMapBooks - Automatic Question Response Grading

## Env configuration
Anaconda env is saved in `requirements.txt` file.

## How to run this repo?

Start`test-server.py` and use `onj-eval.py` script to test all 3 models.
Test server is running on port __8080__.

## Pre trained models
This repo contains pre trained models. These models are located in calculated_models directory.
 
## Manual testing model A & B

Run `test.py` to test model A nd B.

## Manual testing model C
You can manually train and test cnn using `cnn.py`.

### Training CCN
- download http://nlp.stanford.edu/data/glove.6B.zip
- extract .zip file
- copy glove.6B.100d.txt to /input directory

Run main() in cnn.py. This will train cnn, save trained model and run tests. 

### Testing CNN

If you want to test pre calculated model comment line which calls cnn.create_cnn_model(new_model_file_name). Set use_best_model to True to test best calculated model.