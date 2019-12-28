# imageclassifier
An AI program to classifier images to their respective categories (102 outputs) using pre-trained models and Pytorch.

This project is divided into two sections, 
- A train section
- A predict section

The train section of the program is used to train our model. below are the various commands you could use to with this section
- basic usage `python train.py data_directory` . The data_directory must contain at least a `train` and `test` directory. the train directory is used to train the model while the test directory is used to test the model as we train it. This is to enable us visualize the `error loss` and model `accuracy` while we train our model. Giving us an idea on how we could better fine tune our model.
- Setting the checkpoint directory `python train.py data_dir --save_dir save_directory`
- Choose a particular architecture with which you want your model to be trained with `python train.py data_dir --arch "vgg13"`
- Setting the hyperparameters for fine turning `python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20`
- Using the GPU `python train.py data_dir --gpu`

The predict section involves feeding our trained model with data it has never seen to see if it can do the right predictions
- Basic usage `python predict.py /path/to/image checkpoint`
- Return top k most likely categories `python predict.py input checkpoint --top_k 3`
- Using a mapping of categories to real names `python predict.py input checkpoint --category_names cat_to_name.json`
- Using a GPU `python predict.py input checkpoint --gpu`


Enjoy...
