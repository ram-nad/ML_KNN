# ML_KNN
----------
### Implementation of K Nearest Neighbors Classifier in C++.
Implemented Two Feature Scaling:
  - MinMaxScaler 
  - Standard Scaler

Also added a function that measures accuracy in terms of correct predictions as percentage of total predictions.

Supports 3 Distance Metrics:
1. Euclidean
2. Manhattan
3. Chebyshev

Also included a header containg IRIS Flower Data Set.

Added test.cpp: Tests the KNN using IRIS Data Set.

To run cd into directory,

`g++ -c knn.cpp`

`g++ -c test.cpp`

`g++ -o knntest test.o knn.o`

`./knntest`
