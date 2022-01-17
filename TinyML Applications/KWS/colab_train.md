Skip to main content
Training Keyword Spotting in Colab

The code for training your model in Colab is very similar to the code you used to test out the pre-trained model. However, we’d like to point out a couple of things that will hopefully make training in Colab easier (and more intuitive).

Timeouts

After 12 hours your Colab instance will automatically recycle so make sure to download anything you want to keep before then. Also after about 90 minutes of no activity Colab may also recycle the instance as it may deem it “stale.” Therefore, during training it is best practice to check in every hour and click around on the Colab to make sure it knows you are still there! But/and beyond that you can absolutely minimize the window and continue with other work.

File Structure

The training process will create three directories of note.

/data:  will hold all of the data from the Speech Commands dataset. If you want to take a look at particular audio files used for training this is where you can inspect and download them. We will revisit this in Course 3 when you design your own dataset as you’ll need to ensure that it matches this format / file structure.
/train: will hold all of the model checkpoint files. These files describe the state of the model after X steps of training (usually denoted in the file name and the training script auto-saves them every 100 or so steps). If you e.g., trained for 100 steps and wanted to train for another 100 more, you can use the latest checkpoint file at step 100 to start from there instead of re-training from scratch for 200 steps.
/model: will hold all of the final processed model files. There you can find both the frozen Tensorflow model, the TensorflowLite model, and in course 3 you will also find the TensorFlowLite Micro model there. If you’d like to download and save your trained model to compare against another trained model at another time you should make sure to download it from there!
The Training Script

The training script is designed to automate a bunch of the training process. It first sets up the optimizer using a sparse_softmax_cross_entropy loss function. It then runs the optimizer (by default stochastic gradient descent) on the training data and automatically preprocesses the audio files as we’ve discussed in previous sections, such as computing the spectrogram and then the MFCC of the audio stream, before passing them to the model/optimizer.

If you want to take a look at the source code it can be found here: 

https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/speech_commands/train.py. You’ll notice that there are even more flags than we set in the colab. If you’d like to customize the training process even more you can!


Skip to main content
Monitoring Training in Colab

While the Keyword Spotting training script is running (for ~2 hours with the default settings and a GPU runtime), you may want to check-in and analyze how the training is performing. We’ve included two ways that you can do this into the Colab.

TensorBoard

TensorBoard is an application that is designed to help you visualize the training process. It will output graphs of the accuracy over time that look something like the below (this is a screenshot of the staff training a Keyword Spotting model):

A Tensorboard plot showing the training of a Keyword Spotting model. It learns a lot in the first couple thousand iterations and then slowly converges after that

If you’d like to learn more about TensorBoard, Google has put together a nice intro Colab that explore some of its features:

https://colab.research.google.com/github/tensorflow/tensorboard/blob/master/docs/get_started.ipynb

DEBUG Mode

Unfortunately, the staff has found that it sometimes doesn't start showing data for a while (~15 minutes) and sometimes doesn't show data until training completes (and instead shows No dashboards are active for the current data set.). Therefore, we have also set the training script to run in DEBUG mode. By doing so it will print out information about the training process as it progresses. In general you will see printouts at each training step showing the current accuracy that look something like this:

A screenshot of the training results for steps 25-32 of training a keyword spotting model. Where the accuracy bounces around between 31% and 42%.

Remember because we are using a variant of stochastic gradient descent, it is natural for the error to go up and down a bit during training. The key thing to look for is that the larger trend is for the accuracy to be going up on average! For example, if I scroll down a bit on this training run we can see that ~500 steps later we now have much higher accuracy on average (so learning seems to be going well):

A screenshot of the training results for steps 552-560 of training a keyword spotting model. Where the accuracy bounces around between 70% and 83%.

Every 1,000 training steps you will also see a Confusion Matrix printed out that looks something like this:

Image of a confusion matrix for step 1,000 of training a Keyword Spotting Model showing large values along the diagonal and small off-diagonal values indicating that training is working well.

What this is describing is the number of samples that were correctly and incorrectly classified by class for the validation set at this point in the training process. Labeling the axis we can see how this works a little better:

Image of a theoretical confusion matrix between 4 items showing large values along the diagonal and small off-diagonal values indicating that training is working well.

The column plots the predicted item label by the model while the row plots the actual correct label. Down the diagonal in green are the number correctly labeled items. Off the diagonal we find the number of items incorrectly labeled as the various labels. For all of your Keyword Spotting models Item A is “silence”, Item B is “unknown” and then Items C, D, etc. are the various words you chose to train on in the order in which you define them in the Colab. For example in the pretrained model we had:

WANTED_WORDS = "yes,no"

Which means that Item C was “yes” and Item D was “no” and the matrix was a 4x4 matrix. If we had instead set:

WANTED_WORDS = "yes,no,left"

Then it would be a 5x5 matrix and Item C would still be “yes” and Item D would still be “no” and then there would be an Item E that was “left.”

We hope that between TensorBoard and the DEBUG output you’ll be able to understand how well your training is going and gain insights to improve your results!

