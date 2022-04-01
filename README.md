# Identifying Higgs Boson 

Physicists regard the Higgs boson to be the cornerstone of matter's basic structure. This particle is responsible for the mass of all other particles in our universe. The Higgs boson, on the other hand, is extremely difficult to witness directly. Indeed, it only emerges for a fraction of a second in high-speed particle collisions. Fortunately, the decay signature of the boson may be used to infer its identity.
The Higs Boson was discovered at the Large Hadron Collider (LHC) in 2013, over 50 years after its existence was postulated in 1964. Machine learning can assist distinguish whether a signature is due to a Higg Boson or another collision background event since a collision event produces many identical decay signatures. The performance of machine learning models trained on original CERN data in locating the Higg Boson in a collision event will be investigated and evaluated in this research.

All the code we used toward our prediction model is in the github repository as well as our report in which we describe all the different steps of our project.

<object data="https://github.com/CS-433/ml-project-1-ccd_ml/blob/main/Machine_Learning_to_discover_Higgs_Boson.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="https://github.com/CS-433/ml-project-1-ccd_ml/blob/main/Machine_Learning_to_discover_Higgs_Boson.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="https://github.com/CS-433/ml-project-1-ccd_ml/blob/main/Machine_Learning_to_discover_Higgs_Boson.pdf">Download PDF</a>.</p>
    </embed>
</object>

Our best prediction was obtained using ridge regression and the model had an overall accuracy of 0.804.

## Installation

1.You can download the code to run the models from the github repository [Github](https://github.com/CS-433/ml-project-1-ccd_ml.git)


Clone the repository:
```bash
git clone https://github.com/CS-433/ml-project-1-ccd_ml.git
```
2.Unzip the train.csv in the data directory. Once in the right directory, type the following line:

```bash
Unzip train.zip
```
3.Unzip the test.csv in the data directory. Once in the right directory, type the following line:

```bash
Unzip test.zip
```
## Running the program
To run the program, you have to open a terminal on jupyter notebook or an anaconda prompt on anaconda. You have to go in the project repository (in the folder scripts) and run the following command :
```bash
python run.py
```
But be careful, if you have a mac you will have to change path in the run.py (you just have to take the one in comment).
### What will happen
The best method that we used for our prediction, Ridge regression with lambda = 0.0001 and degree = 12, will run and a file sample-submission.csv will be created in the folder data. This file will contain the prediction and can be submit on AIcrowd for testing.


## Methods and parameters
 The different method we used are:
```bash
1    Least Squares
2    Least Squares Gradient Descent
3    Least Squares Stochastic Gradient Descent
4    Ridge Regression
5    Logistic Regression
6    Regularized Logistic Regression
```
The parameters encountered in the different methods are the following.

```bash
Max_iters=maximum number of iterations
Lambda=regularization factor
Gamma=Learning Rate for gradient descents
Batch_size=the size of the batch used for stochastic gradient descent
Degree=degree for feature expansion
```
## Code content
Each method has its corresponding .py file in which the specific functions needed to run a method are coded.

Functions useful to multiple methods such as computing the accuracy, splitting the data to do cross validation are contained in the file useful_fonctions.py (https://github.com/CS-433/ml-project-1-ccd_ml/blob/main/scripts/useful_functions.py) and some already given functions are contained in the projet1_helper.py file (https://github.com/CS-433/ml-project-1-ccd_ml/blob/main/scripts/proj1_helpers.py).

All the functions we used to do our data cleaning and pre-processing are in the file preprocessing.py (https://github.com/CS-433/ml-project-1-ccd_ml/blob/main/scripts/preprocessing.py).

Finally we have an implementations.py file with all methods and there optimized hyperparameters (https://github.com/CS-433/ml-project-1-ccd_ml/blob/main/scripts/implementations.py). 

All those methods can be ran with the run.py file (https://github.com/CS-433/ml-project-1-ccd_ml/blob/main/scripts/run.py). 

