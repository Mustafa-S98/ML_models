from pandas import read_csv as read
from pandas.plotting import scatter_matrix as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as data_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression as log_reg
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read(url, names = names)

array = dataset.values
x = array[:, 0:4]
y = array[:, 4]

x_train, x_validation, y_train, y_validation = data_split(x, y, test_size=0.25, random_state=1)

models = []

models.append(('LR', log_reg(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))


results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print(name, cv_results.mean(), cv_results.std())

final_model = SVC(gamma = 'auto')
model.fit(x_train, y_train)
pred = model.predict(x_validation)
print("---x---")
print(accuracy_score(y_validation, pred))
print()
print(confusion_matrix(y_validation, pred))
print()
print(classification_report(y_validation, pred))