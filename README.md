

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






