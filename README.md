

Train a recurrent network to generate grayscale images using Keras in Python 3.6

One set of weights included, from training over a set of 46 Japanese inkwash (sumi-e) artworks. Training dataset not included.

Easily adapt to train on your own grayscale image set and generate new images. Recommend training on a GPU.

In order to generate images using my pre-trained model, you'll only need the weights-improvement-00-0.4125-3.hdf5 and sample.py files. You'll need to supply sample.py with at least one grayscale image of your own from which the prediction model it can generate a seed. Enter the path to this image or a folder of images in line 19.


## Acknowledgments

* Manuel Garrido and his [keras_monet project](https://github.com/manugarri/keras_monet)
* Jason Brownlee's [tutorial for text generation with LSTMs in Keras](https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/)


