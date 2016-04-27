* MNIST via TensorFlow

** Usage
```
python train.py
```

** Performance
|Name         |                       |
|-------------|-----------------------|
|Accuracy     | 99.27%                |
|Examples/Sec | 10k                   |

** Environment
|Name      |            Description|
|----------|-----------------------|
|GPU       | GeForce GTX TITAN X   |
|Accuracy  | 99.27%                |
|OS        | Ubuntu 16.04 LTS      |
|Library   | TensorFlow 0.8.0      |

** Network Architecture
|Layer Type   | Parameters                         |
|-------------|------------------------------------|
|input        | size:32x32, channels:1             |
|convolution  | kernel:3x3, channels:32, padding:1 |
|relu         |                                    |
|max pooling  | kernel:2x2, strides: 2             |
|convolution  | kernel:3x3, channels:64, padding:1 |
|relu         |                                    |
|max pooling  | kernel:2x2, strides: 2             |
|dropout      | rate: 0.5                          |
|linear       | channels: 1024                     |
|relu         |                                    |
|linear       | channels: 10                       |
|relu         |                                    |
|softmax      |                                    |
