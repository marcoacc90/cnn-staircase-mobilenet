# cnn-staircase-mobilenet
MobileNet for staircase detection

# Installing prerequisites
Python 3.6.9
Tensorflow 2.1.0
install matplotlib
install pillow

# Training the system
python3 MobileNetTrain.py > <output>

  (1) Configure the number of epochs in Global.py, default: 500.
  (2) <output>: training evolution (suggested name: models/mobilenet_train_evo_raw.txt).

# Test the system
python3 MobileNetTest.py >  <output>

  (1) Select the model to be tested by selecting the number of epochs in Global.py, default: 500.
  (2) <output>: labels | execution time (suggested name: models/mobilenet_500_test.txt).

# Results
     - Images with labels

     python3 ObstacleDetection.py

     (1) Select the model by selecting the number of epochs in Global.py, default: 500.
     (2) The model should be ready, models/mobilenet_weights_500.
     (3) The labels should be prepared, models/mobilenet_500_test.txt.
     (4) Output: a folder with the images, (output folder: test_mobilenet_500/)

     - Metrics (scripts/)

     Execute the Metrics.m script,

     (1) Open the Metrics.m script.
     (2) Select the mode: train, validation, or test.
     (3) Select the model by selecting the number of epochs, default: 500.
     (4) The model should be ready, models/mobilenet_weights_500.
     (5) The labels should be prepared, models/mobilenet_500_test.txt.
     (6) Output: the accuracy and other measures in the selected dataset (train, validation, or test).

     - Training evolution (scripts/)

     (1) Execute the script formatTrainEvolution.py to format the training evolution (raw data) into the format: epoch | loss_train | acc_train | loss_validation | acc_validation.

     python3 formatTrainEvolution <raw_train_evolution> > <train_evolution>

     <raw_train_data>: ./../models/mobilenet_train_evo_raw.txt
     <train_evolution>: ./../models/mobilenet_train_evo.txt

     (2) Visualize the training evolution (PlotTrainEvolution.m),
          Select the name of the file.
          Output: acc and loss visualizations.
