#include<cstddef>
#include<cmath>

namespace knn{

enum class Distance{
	EUCLIDEAN,
	MANHATTAN,
	CHEBYSHEV
};

/* Euclidean Distance = sqrt(sum((pt1 - pt2)^2)) */
double euclidean_distance(const double *pt1, const double *pt2, size_t feature_count){
	double sum = 0;
	double diff;
	for(size_t i=0; i<feature_count; i++){
		diff = pt1[i] - pt2[i];
		sum += diff*diff;
	}
	sum = sqrt(sum);
	return sum;
}

/* Manhattan Distance = sum(|pt1 - pt2|) */
double manhattan_distance(const double *pt1, const double *pt2, size_t feature_count){
	double sum = 0;
	double diff;
	for(size_t i=0; i<feature_count; i++){
		diff = abs(pt1[i] - pt2[i]);
		sum += diff;
	}
	return sum;
}

/* Chebyshev Distance = max(|pt1 - pt2|) */
double chebyshev_distance(const double *pt1, const double *pt2, size_t feature_count){
	double sum = 0;
	double diff;
	for(size_t i=0; i<feature_count; i++){
		diff = abs(pt1[i] - pt2[i]);
		if(sum < diff)
			sum = diff;
	}
	return sum;
}

/* Class for storing distance and label for k nearest neighbors */
class data{
	double distance;
	int label;
	public:
	data();
	data(double dist, int l);
	bool operator>(const data &obj) const;
	bool operator<(const data &obj) const;
	bool operator>=(const data &obj) const;
	bool operator<=(const data &obj) const;
	data &operator=(const data &obj);
	int get_label() const;
	double get_dis() const;
};

data::data(): distance{1e37}{
}
data::data(double dist, int l): distance{dist}, label{l}{
}
int data::get_label() const{
	return this->label;
}
double data::get_dis() const{
	return this->distance;
}
bool data::operator>(const data &obj) const{
	return (this->distance > obj.distance);
}
bool data::operator<(const data &obj) const{
	return (this->distance < obj.distance);
}
bool data::operator>=(const data &obj) const{
	return (this->distance >= obj.distance);
}
bool data::operator<=(const data &obj) const{
	return (this->distance <= obj.distance);
}
data &data::operator=(const data &obj){
	this->distance = obj.distance;
	this->label = obj.label;
	return *this;
}

/* Function for storing only k nearest neighbors's data */

/* Uses Max-Heap */

void insert_neighbor(data *arr, size_t lim, const data &value){
	if(value > arr[0])
		return; // If distance is greater than maximum then do nothing.
	else
		arr[0] = value; // Else replace maximum distance with this value.
	size_t max = 0;
	size_t node;
	size_t left;
	size_t right;
	data temp;
	while(true){ // While correct position in Heap is not found exchange elements.
		node = max;
		left = 2*node + 1;
		right = 2*node + 2;
		if(left < lim && arr[left] > arr[node])
			max = left;
		else
			max = node;
		if(right < lim && arr[right] > arr[max])
			max = right;
		if(node != max){
			temp = arr[max];
			arr[max] = arr[node];
			arr[node] = temp;
		}
		else{
			break;
		}
	}
}

/* Classifier for kNN */
class Classifier{
	size_t k_neighbors;
	data *ptr;
	size_t n_points;
	size_t n_features;
	double **Attributes;
	int *labels;
	void delete_model();
	double (*distance_func)(const double *, const double *, size_t feature_count);
	public:
	Classifier() = delete; // Default Constructor not allowed
	Classifier(const Classifier& obj) = delete; // Copy Construction not allowed
	Classifier& operator=(const Classifier& obj) = delete; // Assignment not allowed
	explicit Classifier(size_t n, Distance dis=Distance::EUCLIDEAN);
	~Classifier();
	void fit(size_t ncount, size_t fcount, const double* const* train_data, const int *train_label);
	int predict(const double *attr);
};

/* Constructor: Initializes value for k and the Distance Metric */
Classifier::Classifier(size_t n, Distance dis): k_neighbors{0}, ptr{nullptr}, Attributes{nullptr}, labels{nullptr}{
	if(n <= 0){
		return; // If n is invalid return
	}
	if(dis == Distance::EUCLIDEAN){
		distance_func = euclidean_distance;
	}
	else if(dis == Distance::MANHATTAN){
		distance_func = manhattan_distance;
	}
	else if(dis == Distance::CHEBYSHEV){
		distance_func = chebyshev_distance;
	}
	else{
		return; // If distance metric is invalid then return
	}
	k_neighbors = n;
	ptr = new data[k_neighbors];
}

/* Destructor: Deletes the model and data */
Classifier::~Classifier(){
	delete_model();
	if(ptr != nullptr)
		delete[] ptr;
}

/* Function for deleting data points and their labels */
void Classifier::delete_model(){
	if(labels != nullptr){
		delete[] labels;
	}
	if(Attributes != nullptr){
		for(int i=0; i<n_points; i++){
			delete[] Attributes[i];
		}
		delete[] Attributes;
	}
}

/* Train Classifier with new datasets */
void Classifier::fit(size_t ncount, size_t fcount, const double* const* train_data, const int *train_label){
	if(ncount <= 0 || fcount <= 0)
		return;	
	if(Attributes != nullptr || labels != nullptr){
		delete_model();
	}
	n_points = ncount;
	n_features = fcount;
	labels = new int[n_points];
	Attributes = new double*[n_points];
	for(size_t i=0; i<n_points; i++){
		labels[i] = train_label[i];
		Attributes[i] = new double[n_features];
		for(size_t j=0; j<n_features; j++){
			Attributes[i][j] = train_data[i][j];
		}
	}
}

int Classifier::predict(const double *attr){
	data empty;
	for(size_t i=0; i<k_neighbors; i++){
		ptr[i] = empty; // Make Heap Empty
	}
	for(size_t i=0; i<n_points; i++){ // Calculate distance from all points and store only k nearest of them
		data value(distance_func(attr, Attributes[i], n_features), labels[i]);
		insert_neighbor(ptr,k_neighbors,value);
	}
	size_t max_count = 0;
	size_t cur_count;
	int prediction = empty.get_label();
	int cur_prediction;
	for(size_t i=0; i<k_neighbors; i++){ // Make a current prediction and count it's occurance
		cur_prediction = ptr[i].get_label();
		cur_count = 0;
		for(size_t j=0; j<k_neighbors; j++){
			if(cur_prediction == ptr[j].get_label())
				cur_count++;
		}
		if(cur_count > max_count){ // If current prediction has maximum occurance then store it in prediction
			max_count = cur_count;
			prediction = cur_prediction;
		}
	}
	return prediction;
}


/* 
HOW MIN-MAX SCALER WORKS:
It transforms input values to fit the range [min,max]
Minimum element in input gets mapped to min
Maximum element in input gets mapped to max
output[i] = (input[i] - shift)/scale
scale = (MAX - MIN)/(max-min)
shift = MIN - min*scale (Can also use shift = MAX - max*scale)
MAX is maximum value in input
MIN is minimum value in input
*/

void MinMaxScaler(double *input, size_t len, double min=0, double max=1, double *output=nullptr){
	/* If len(length of input array), min(minimum value after scaling), 
	   max(maximum value after scaling) are wrong then return without any processing */
	if(len <= 0)
		return;
	if(min >= max)
		return;
	if(output == nullptr)
		output = input; // If output is default change it to be same as input.
	double MAX = input[0]; // MAX is assumed to be first element
	double MIN = input[0]; // MIN is assumed to be first element
	double scale, shift;
	for(size_t i=1; i<len; i++){
		if(input[i] > MAX)
			MAX = input[i];
		if(input[i] < MIN)
			MIN = input[i];
	}
	scale = (MAX - MIN)/(max - min); // Compute scale value
	shift = (MIN - (min*scale)); // Compute shift value
	for(size_t i=0; i<len; i++){
		output[i] = (input[i] - shift)/scale;
	}
}

/*
HOW STANDARD SCALER WORKS:
It transforms input values to sequence having zero mean and unit variance
output[i] = (input[i] - mean)/stndv
mean = average of input values
stndv = standard deviation of input values
*/

void StandardScaler(double *input, size_t len, double *output=nullptr){
	/* If len(length of input array) is wrong then return */
	if(len <= 0)
		return;
	if(output == nullptr)
		output = input; // If output is default change it to be same as input.
	double mean = 0; // Initialize mean to 0
	for(size_t i=0; i<len; i++){
		mean += (input[i]/len); // Adding this way ensures that sum does not exceed limits
	}
	double variance = 0;
	double diff;
	for(size_t i=0; i<len; i++){
		diff = input[i] - mean;
		variance += (diff*diff)/len; // Compute variance
	}
	double stndv = sqrt(variance);
	if(stndv == 0.0){
		for(size_t i=0; i<len; i++){
			output[i] = 0;
		}
	}
	else{
		for(size_t i=0; i<len; i++){
			output[i] = (input[i] - mean)/stndv;
		}
	} 
}

/*
Measures Accuracy as percentage of correct predictions made by Model.
*/

double accuracy(const int *prediction, const int *expected, size_t size){
	if(size <= 0)
		return 0.0;
	double sum = 0;
	for(size_t i=0; i<size; i++){
		sum += ((prediction[i] == expected[i])?(1.0):(0.0)); // Add one if prediction matches expected value
	}
	sum = (sum*100.0)/size;
	return sum;
}

}
