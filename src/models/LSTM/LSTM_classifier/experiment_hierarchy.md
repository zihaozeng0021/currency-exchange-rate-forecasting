```mermaid
graph TD
    1[1, one layer LSTM, sigmoid activation, binary_crossentropy loss, RMSprop optimizer] --> 2[2, + MinMax Scaling]
    1 --> 3[3, + Log transformation]
    1 --> 4[4, + Z-score Scaling]
    2 --> 5[5, + smoothing data using moving average]
    2 --> 6[6, + smoothing data using exponential moving average]
    2 --> 7[7, + smoothing data using differencing]
    3 --> 8[8, + smoothing data using moving average]
    3 --> 9[9, + smoothing data using exponential moving average]
    3 --> 10[10, + smoothing data using differencing, X]
    7 --> 11[11, + sigmoid activation function, currently the baseline]
    7 --> 12[12, + hard sigmoid activation function]
    7 --> 13[13, + linear activation function, default on keras]
    11 --> 14[14, + binary_crossentropy loss, currently the baseline]
    11 --> 15[15, + binary_focal_crossentropy loss]
    11 --> 16[16, + poisson loss]
    11 --> 17[17, + kl_divergence loss]
    15 --> 18[18, + SGD optimizer]
    15 --> 19[19, + RMSprop optimizer, default on Keras, currently the baseline]
    15 --> 20[20, + Adam optimizer]
    15 --> 21[21, + Adamax optimizer]
    15 --> 22[22, + Nadam optimizer]
    15 --> 23[23, + Adgrad optimizer]
    15 --> 24[24, + Adadelta optimizer]
    15 --> 25[25, + Ftrl optimizer]
    22 --> 26[26, + Fixed threshold, currently the baseline]
    22 --> 27[27, + Grid search for threshold]
    22 --> 28[28, + ROC curve]
    22 --> 29[29, + Precision-Recall curve]
    22 --> 30[30, + Cost-Sensitive Thresholding]
    22 --> 31[31, + Adaptive Thresholding]
    29 --> 32[Applied best parameters]
    
    
    
    
    classDef brightBlue fill:#00BFFF,stroke:#333,stroke-width:2px;
    class 2,3,4,5,6,7,8,9,10 brightBlue;
    
    classDef brightGreen fill:#00FF00,stroke:#333,stroke-width:2px;
    class 11,12,13 brightGreen;
    
    classDef brightRed fill:#FF0000,stroke:#333,stroke-width:2px;
    class 14,15,16,17 brightRed;
    
    classDef brightYellow fill:#FFFF00,stroke:#333,stroke-width:2px;
    class 18,19,20,21,22,23,24,25 brightYellow;
    
    classDef brightOrange fill:#FFA500,stroke:#333,stroke-width:2px;
    class 26,27,28,29,30,31 brightOrange;

    
    
```