# HSTree - Anomaly Detection For Streaming Data

## Introduction

Half-Space Tree, known as HSTree, is an efficient data structure for detecting anomalies in streaming data. It is robust when incoming data is endless and contains a large amount of normal data. HSTree has a constant amortised time complexity and a constant memory requirement when processing data. It also has a robust detection accuracy when parameters are different. 

This repository contains my implementation of HSTree data structure and tests with HTML and SMTP data. The implementation is in Python3 and the idea of HSTree is based on the Tan, Ting and Liu's research[1].

## Setup
### Prequisites
1. `Python (>=3.3)`

This script is written in Python3 style. Please make sure the latest Python3 is installed on your computer before the next steps. If you already have Python installed, please make sure its version should be at least *3.3*.

2. `NumPy (>= 1.8.2)` and `SciPy (>= 0.13.3)`

`NumPy` is the fundamental package for scientific computing with Python and `SciPy` has a collection of numerical algorithms. If you don't have `NumPy` and `SciPy` installed, please use the following command to install them.[2]

~~~~
python3 -m pip install --user numpy scipy
~~~~

3. `scikit-learn`

This script used the `MinMaxScaler` module of `Scikit-learn`. Please also install this package using the following command.[3]

~~~
pip install scikit-learn
~~~

4. Other packages

If you miss the prequisites of the packages listed in 1 - 3, please follow the instructions to install them.

## Implementation Details
### File Structure
1. `HS tree.py`: The implementation of HSTree with initialization of test cases.
2. `http.csv` and `labels.csv`: The sample HTTP data and its corresponding labels.
3. `smtp.csv` and `smtp_labels.csv`: The sample SMTP data and its corresponding labels.

The corresponding label files are only used for test the accuracy of anomaly detection.

### Core Functions
#### 1. `BuildSingleHSTree(max_arr, min_arr, k, h, dimensions)`
The method `BuildSingleHSTree` is used to build one single `HSTree`. It returns a value of `Node` class and requires the following five input arguments:

`max_arr`: An array containing the maximum values of each dimension.

`min_arr`: An array containing the minimum values of each dimension.

`k`: The depth of the current node.

`h`: The maximum depth value.

`dimensions`: The number of dimensions.

#### 2. `UpdateMass(x, node, ref_window)`
This method updates the mass profile of normal data in each node. It takes the following three input arguments:

`x` : An instance of data.

`node` : The node in an HSTree.

`ref_window` : Boolean value. If the instance `x` is in current reference window, then it is `true`. Otherwise `ref_window` is set to `false`.

#### 3. `StreamingHSTrees(X, psi, t, h)`
This method is the major sequence of evaluating anomalies in streaing data. It takes four arguments, `X`, `psi`, `t` and `h`, as inputs and returns a `list` of scores of the instances. This method can be called directly for test data. The input arguments are specified as the following:

 `X`: The input data. This implementation only simulates the data streaming process, so all data is provided at the beginning.

`psi`: Window Size. The HSTrees are initialized with the first `psi` instances. In this implementation, the first `psi` elements in `X` will be the initial data of HSTrees.

`t`: Number of HSTrees. 

`h`: The maximum depth of each HSTree.

#### 4. `accuracy_value(scores, y, num)`
The method `accuracy_value` is a helper method to evaluate the test results. It prints four values: `True Positive`, `False Positive`, `True Negative` and `False Negative`. The inputs are:

`scores`: A `List` of scores. It should be the return value of `StreamingHSTrees()` method.

`y`: A `List` of reference labels (Ground truth).

`num`: The total number of anomalies in the dataset. This implementation is not elegant. I will refactor it in the future.

## Test

To test this data structure, you need to do the following steps.

### 1. Generate Input Sequence
The input sequence can be hard-coded or loaded from files. Please initialize it with Numpy. Here are two examples.

~~~~python
# Initialize with hard-coded data
X = [[0.5], [0.45], [0.43], [0.44], [0.445], [0.45], [0.0]]
X = np.array(X)

# Load from file
X = np.genfromtxt('http.csv',delimiter=',')
~~~~

### 2. Data Preprocessing

If the data is not distributed in 0-1, we need to preprocess the data to satisfy this property.

~~~~python
# Use MinMaxScaler in scikit-learn to preprocess the data 
scaler = MinMaxScaler()
X_new = scaler.fit_transform(X)
~~~~

### 3. Simulation of Data Streaming

Call the method `StreamingHSTrees` to simulate data streaming process and get a list of scores of the evaluated instances. You may need to adjust the parameters to get a better accuracy, but HSTree is not sensitive to parameters.

~~~~python
# Parameters need to adjust
final_scores = StreamingHSTrees(X_new, psi, t, h)
~~~~

### 4. Evaluation

Call `accuracy_value` to evaluate the anomaly detection in step 3.

~~~~python
# "y" is the reference labels and "13" is the number of the anomalies.
accuracy_value(final_scores, y, 13)
~~~~

## Future Work

This implementation of HSTree only simulates its procedures. It can be modified and extended to receive real streaming data. The script will be able to read data instances dynamically from text files or real Internet streams.

## Reference
1. Fast Anomaly Detection for Streaming Data (Tan, Ting, Liu) [https://pdfs.semanticscholar.org/73b6/b7d9e7e225719ad86234927a3b60a4a873c0.pdf](https://pdfs.semanticscholar.org/73b6/b7d9e7e225719ad86234927a3b60a4a873c0.pdf)

2. SciPy Official Site [https://www.scipy.org/about.html](https://www.scipy.org/about.html)

3. Installing scikit-learn [http://scikit-learn.org/stable/install.html](http://scikit-learn.org/stable/install.html)
