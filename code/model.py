import numpy as np
from tqdm import tqdm


class PCA:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.components = None
    
    def fit(self, X) -> None:
        # calculate the mean of the data
        self.mean = np.mean(X, axis=0)
        
        # center the data
        # X_centered = X - self.mean
        
        # calculate the covariance matrix
        covariance_matrix = np.cov(X.T)
        
        # calculate the eigenvectors and eigenvalues of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        
        # sort the eigenvectors by eigenvalues in descending order
        eigenvectors = eigenvectors.T
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[sorted_indices]
        
        # select the first n_components eigenvectors as the components
        self.components = sorted_eigenvectors[:self.n_components]
    
    def transform(self, X) -> np.ndarray:
        # center the data
        X_centered = X - self.mean
        
        # project the data onto the components
        return np.dot(X_centered, self.components.T)

    def fit_transform(self, X) -> np.ndarray:
        # fit the model and transform the data
        self.fit(X)
        return self.transform(X)


class SupportVectorModel:
    def __init__(self) -> None:
        self.w = None
        self.b = None
    
    def _initialize(self, X) -> None:
        # initialize the parameters
        self.w = np.zeros(X.shape[1])
        self.b = 0

    def fit(
            self, X, y, 
            learning_rate: float,
            num_iters: int,
            C: float = 1.0,
    ) -> None:
        self._initialize(X)
        
        # fit the SVM model using stochastic gradient descent
        for i in tqdm(range(1, num_iters + 1)):
            # sample a random training example
            idx = np.random.randint(X.shape[0])
            x_i, y_i = X[idx], y[idx]
            
            # compute the hinge loss and gradient
            loss = max(0, 1 - y_i * (np.dot(x_i, self.w) + self.b))
            grad_w = (-1* C * y_i * x_i) if loss > 0 else 0
            grad_b = (-1* C * y_i) if loss > 0 else 0
            
            # update the parameters
            self.w -= learning_rate * (grad_w + self.w / num_iters)
            self.b -= learning_rate * grad_b
        
    
    def predict(self, X) -> np.ndarray:
        # make predictions for the given data
        return np.sign(np.dot(X, self.w) + self.b)

    def accuracy_score(self, X, y) -> float:
        # compute the accuracy of the model (for debugging purposes)
        return np.mean(self.predict(X) == y)


class MultiClassSVM:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.models = []
        # class_label=2
        for i in range(self.num_classes):
            self.models.append(SupportVectorModel())
    
    def fit(self, X, y, **kwargs) -> None:
        # first preprocess the data to make it suitable for the 1-vs-rest SVM model
        # then train the 10 SVM models using the preprocessed data for each class
        for i in range(self.num_classes):
            y_i = np.where(y == i, 1, -1)
            self.models[i].fit(X, y_i, **kwargs)

    def predict(self, X) -> np.ndarray:
        # pass the data through all the 10 SVM models and return the class with the highest score
        scores = np.zeros((X.shape[0], self.num_classes))
        for i in range(self.num_classes):
            scores[:, i] = self.models[i].predict(X)
        return np.argmax(scores, axis=1)

    def accuracy_score(self, X, y) -> float:
        return np.mean(self.predict(X) == y)
    
    def precision_score(self, X, y) -> float:   
        preds = self.predict(X)
        prec = []

        for class_lable in range(self.num_classes):
            tp = np.sum((preds == class_lable) & (y == class_lable))
            false_pos = np.sum((preds == class_lable) & (y != class_lable))

            if tp+false_pos == 0:
                prec.append(0)

            else:
                prec.append(tp / (tp + false_pos))

        return np.mean(prec)


    def recall_score(self, X, y) -> float:
        preds = self.predict(X)
        recall = []

        for class_lable in range(self.num_classes):
            tp = np.sum((preds == class_lable) & (y == class_lable))
            fn = np.sum((preds != class_lable) & (y == class_lable))

            if tp+fn==0:
                recall.append(0)

            else:
                recall.append(tp / (tp + fn))

        return np.mean(recall)

    
    def f1_score(self, X, y) -> float:
        precision = self.precision_score( X, y)
        recall = self.recall_score( X, y)
        if precision + recall == 0:
            return 0.0
        return (2 * (precision * recall) / (precision + recall))    



    

    
