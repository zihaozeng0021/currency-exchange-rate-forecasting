```mermaid
graph TD
    1[1, one layer LSTM, linear activation, mse loss, RMSprop optimizer] --> 2[2, + MinMax Scaling]
    1 --> 3[3, + Log transformation]
    1 --> 4[4, + Z-score Scaling]
    4 --> 5[5, + linear activation function, currently the baseline and default on keras]
    4 --> 6[6, + exponential activation function]
    4 --> 7[7, + softplus activation function]
    5 --> 8[8, + mse loss, currently the baseline]
    5 --> 9[9, + mean_absolute_error loss]
    5 --> 10[10, + cosine_similarity loss]
    5 --> 11[11, + huber loss]
    5 --> 12[12, + logcosh loss]
    5 --> 13[13, + mean_squared_logarithmic_error loss]
    8 --> 14[14, + SGD optimizer]
    8 --> 15[15, + RMSprop optimizer, default on Keras, currently the baseline]
    8 --> 16[16, + Adam optimizer]
    8 --> 17[17, + Adamax optimizer]
    8 --> 18[18, + Nadam optimizer]
    8 --> 19[19, + Adamgrad optimizer]
    8 --> 20[20, + Adadelta optimizer]
    8 --> 21[21, + Ftrl optimizer]
    15 --> 22[Best so far]
    
    
    classDef brightBlue fill:#00BFFF,stroke:#333,stroke-width:2px;
    class 2,3,4 brightBlue;
    
    classDef brightGreen fill:#00FF00,stroke:#333,stroke-width:2px;
    class 5,6,7 brightGreen;
    
    classDef brightRed fill:#FF0000,stroke:#333,stroke-width:2px;
    class 8,9,10,11,12,13 brightRed;
    
    classDef brightYellow fill:#FFFF00,stroke:#333,stroke-width:2px;
    class 14,15,16,17,18,19,20,21 brightYellow;
    
```