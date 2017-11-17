

Developed for Tensorflow 1.1

# Structure


Assuming one starts with a 32x32 image, the dimensions at each level
of filtering are as follows:


| Layer |  Op  | KrnSz | Stride(v,h) | OutDim |  H |  W  | PadOpt
|:-----:|------|-------|:-----------:|--------|----|-----|--------------
| 1     | Conv |   3   |   1         |   64   | 30 | 30  |    valid
| 2     | Conv |   3   |   1         |   64   | 30 | 30  |    same
|       | Pool |   2   |   2         |   64   | 15 | 15  | 
| 3     | Conv |   3   |   1         |  128   | 15 | 15  |    same
| 4     | Conv |   3   |   1         |  128   | 15 | 15  |    same
|       | Pool |   2   |   2,1       |  128   |  7 | 14  |       
| 5     | Conv |   3   |   1         |  256   |  7 | 14  |    same
| 6     | Conv |   3   |   1         |  256   |  7 | 14  |    same
|       | Pool |   2   |   2,1       |  256   |  3 | 13  |       
| 7     | Conv |   3   |   1         |  512   |  3 | 13  |    same
| 8     | Conv |   3   |   1         |  512   |  3 | 13  |    same
|       | Pool |   3   |   3,1       |  512   |  1 | 13  |     
| 9     | LSTM |       |             |  512   |    |     |              
| 10    | LSTM |       |             |  512   |    |     |              

To accelerate training, a batch normalization layer is included before
each pooling layer and ReLU non-linearities are used throughout. Other
model details should be easily identifiable in the code.

# Training

To completely train the model, you will need to download the mjsynth
dataset, pack it into sharded tensorflow records. Then you can start
the training process, a tensorboard monitor, and an ongoing evaluation
thread. The individual commands are packaged in the accompanying `Makefile`.

    make mjsynth-download
    make mjsynth-tfrecord
    make train &
    make monitor &
    make test

To monitor training, point your web browser to the url (e.g.,
(http://127.0.1.1:8008)) given by the Tensorboard output.

Note that it may take 4-12 hours to download the complete mjsynth data
set. A very small set (0.1%) of packaged example data is included; to
run the small demo, skip the first two lines involving `mjsynth`.

With a Geforce GTX 1080, the demo takes about 20 minutes for the
validation character error to reach 45% (using the default
parameters); at one hour (roughly 7000 iterations), the validation
error is just over 20%.

With the full training data, by one million iterations the model
typically converges to around 7% training character error and 35% word
error, both varying by 2-5%.




