namespace knn{

/* MinMaxScaler: Transforms input values to fit hte range [min,max] */

void MinMaxScaler(double *input, size_t len, double min=0, double max=1, double *output=nullptr);

/*
input - Values to be transformed.
len - Number of Values.
ouput - Scaled Values.
If no output is specified then by default values of input array changes.
min - Minimum value in range. By default 0.
max - Maximum value in range. By default 1.
*/

/* StandardScaler: Transforms input values to sequence with mean 0, and variance 1. */

void StandardScaler(double *input, size_t len, double *output=nullptr);

/*
input - Values to be transformed.
len - Number of Values.
ouput - Scaled Values.
If no output is specified then by default values of input array changes.
*/

/* Distance Metrics */

enum class Distance{
	EUCLIDEAN, // Euclidean Distance = sqrt(sum((x1 - x2)^2))
	MANHATTAN, // Manhattan Distance = sum(|x1 - x2|)
	CHEBYSHEV // Chebyshev Distance = max(|x1 - x2|)
};

/*
Example Usage: Distance::CHEBYSHEV.
*/


/* Distance Computing Functions */

double euclidean_distance(const double *pt1, const double *pt2, size_t feature_count);
double manhattan_distance(const double *pt1, const double *pt2, size_t feature_count);
double chebyshev_distance(const double *pt1, const double *pt2, size_t feature_count);

/*
pt1: Array of features for first point.
pt2: Array of features for second point.
feature_count: Count of features.
*/

/* Classifier for kNN */

class data;

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

/*
Default Constructor not allowed.
Copy Construction not allowed.
Assignment not allowed.
Constructor: value for k, Distance Metric to be used (Must be of enum class Distance, by default EUCLIDEAN).
If invalid parameters are passed then Classifier is not initialized and it won't work.
fit: Make a new model with specified data points.
	ncount: Number of data points.
	fcount: Number of features.
	train: Data in form of Array of Doubles([point][feature]). There must be ncount data points and fcount features.
	labels: Labels for ncount data points.
	If data is not in specified format then it may result in segmentation fault.
predict: Predict Label of new data point
	attr: Faetures of data point to be predicted. Size must be same as features count in current model.
*/

/* Accuracy Measurement: Percentage of correct predictions made */

double accuracy(const int *prediction, const int *expected, size_t size);

/*
prediction: Array having predicted values.
expected: Array having expected(correct) values.
size: Number of elements in both array.
*/

}
