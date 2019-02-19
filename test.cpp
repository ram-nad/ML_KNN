/*
A Test Program for KNN on IRIS DataSet.
120 Data Points are used for training the KNN.
30 Data are then used for making predictions.
Accuracy is measured.
*/

#include<iostream>
#include "knn.h"
#include "iris.h"
using namespace std;
using knn::Classifier;

int main(){
	Classifier csf(3); // Classifier having k=3 and Distance Metric is Euclidean(default)
	csf.fit(120,4,iris_features,iris_labels); // Train the Model
	int pd[30];
	for(int i=0; i<30; i++){
		pd[i] = csf.predict(iris_features[120+i]); // Store Predictions
	}
	cout << "Prediction:" << endl;
	for(int i=0; i<30; i++){
		cout << pd[i] << " ";
	}
	cout << endl;
	cout << "Expected:" << endl;
	for(int i=0; i<30; i++){
		cout << iris_labels[120+i] << " ";
	}
	cout << endl;
	cout << "Accuracy: " <<  knn::accuracy(pd,iris_labels+120,30) << endl; // Measure Accuracy
	return 0;
}
