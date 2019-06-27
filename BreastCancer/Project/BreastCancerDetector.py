import numpy as np
import numpy as np
import pandas as pd
from tkinter import *
from tkinter.messagebox import *
from sklearn import model_selection
from sklearn.impute import SimpleImputer
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('breast-cancer-wisconsin.csv')
data.drop(['Sample code number'], axis=1, inplace=True)
data.replace('?', 0, inplace=True)

# Convert the DataFrame object into a Numpy array so as to be able to impute
values = data.values

# Now impute
imputer = SimpleImputer()
imputedData = imputer.fit_transform(values)

# Normalize the range of features to uniform range
scaler = MinMaxScaler(feature_range=(0, 1))
normalizeData = scaler.fit_transform(imputedData)

#Segregate features from label
X = imputedData[:, 0:9]
Y = imputedData[:,9]

kfold = model_selection.KFold(n_splits=10, random_state=7)
cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=7)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

# AdaBoost Classification

from sklearn.ensemble import AdaBoostClassifier
seed = 7
num_trees = 70
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

# Voting Ensemble for Classification

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

kfold = model_selection.KFold(n_splits=10, random_state=seed)

# create the sub models
estimators = []
model1 = LogisticRegression(solver='lbfgs', max_iter=10000)
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC(gamma='auto')
estimators.append(('svm', model3))

# create the ensemble model
ensemble = VotingClassifier(estimators)
results = model_selection.cross_val_score(ensemble, X, Y, cv=kfold)
print(results.mean())

# Create prediction window
ensemble.fit(X, Y)


def getvalue():
    ls = [var1.get(), var2.get(), var3.get(), var4.get(), var5.get(), var6.get(), var7.get(), var8.get(), var9.get()]
    feat = np.array(ls).reshape(1, 9)
    pred = ensemble.predict(feat)
    prediction = int(pred[0])

    if prediction == 2:
        print(showinfo("Feedback", "Likely Benign"))
    else:
        print(showinfo("Feedback", "Likely Malignant"))


root = Tk()
root.title("Breast Cancer Prediction")
root.iconbitmap('breast.ico')
windowWidth = 650
windowHeight = 650

screenWidth = root.winfo_screenwidth()
screenHeight = root.winfo_screenheight()

x_coordinate = (screenWidth/2 - windowWidth/2)
y_coordinate = (screenHeight/2 - windowHeight/2)
root.geometry("%dx%d+%d+%d" % (windowWidth, windowHeight, x_coordinate, y_coordinate))

label = Label(root, text='Breast Cancer Predictor', bg='yellow', fg='purple', font=('georgia', 30, 'bold'))
label.grid(columnspan=2)

var1 = IntVar()
var2 = IntVar()
var3 = IntVar()
var4 = IntVar()
var5 = IntVar()
var6 = IntVar()
var7 = IntVar()
var8 = IntVar()
var9 = IntVar()

label1 = Label(root, text='Clump Thickness', font=('georgia', 12, 'bold')).grid(row=1, sticky=E)
label2 = Label(root, text='Uniformity of Cell Size', font=('georgia', 12, 'bold')).grid(row=2, sticky=E)
label3 = Label(root, text='Uniformity of Cell Shape', font=('georgia', 12, 'bold')).grid(row=3, sticky=E)
label4 = Label(root, text='Marginal Adhesion', font=('georgia', 12, 'bold')).grid(row=4, sticky=E)
label5 = Label(root, text='Single Epithelial Cell Size', font=('georgia', 12, 'bold')).grid(row=5, sticky=E)
label6 = Label(root, text='Bare Nuclei', font=('georgia', 12, 'bold')).grid(row=6, sticky=E)
label7 = Label(root, text='Bland Chromatin', font=('georgia', 12, 'bold')).grid(row=7, sticky=E)
label8 = Label(root, text='Normal Nucleoli', font=('georgia', 12, 'bold')).grid(row=8, sticky=E)
label9 = Label(root, text='Mitoses', font=('georgia', 12, 'bold')).grid(row=9, sticky=E)

slider1 = Scale(root, orient=HORIZONTAL, variable=var1, tickinterval=1, length=400, from_=0, to=10).grid(row=1, column=1)
slider2 = Scale(root, orient=HORIZONTAL, variable=var2, tickinterval=1, length=400, from_=0, to=10).grid(row=2, column=1)
slider3 = Scale(root, orient=HORIZONTAL, variable=var3, tickinterval=1, length=400, from_=0, to=10).grid(row=3, column=1)
slider4 = Scale(root, orient=HORIZONTAL, variable=var4, tickinterval=1, length=400, from_=0, to=10).grid(row=4, column=1)
slider5 = Scale(root, orient=HORIZONTAL, variable=var5, tickinterval=1, length=400, from_=0, to=10).grid(row=5, column=1)
slider6 = Scale(root, orient=HORIZONTAL, variable=var6, tickinterval=1, length=400, from_=0, to=10).grid(row=6, column=1)
slider7 = Scale(root, orient=HORIZONTAL, variable=var7, tickinterval=1, length=400, from_=0, to=10).grid(row=7, column=1)
slider8 = Scale(root, orient=HORIZONTAL, variable=var8, tickinterval=1, length=400, from_=0, to=10).grid(row=8, column=1)
slider9 = Scale(root, orient=HORIZONTAL, variable=var9, tickinterval=1, length=400, from_=0, to=10).grid(row=9, column=1)

button1 = Button(root, text="PREDICT",  command=getvalue, bg='#FF8800', font=('arial', 15, 'bold')).grid(row=12, sticky=W)
button2 = Button(root, text="EXIT",  command=root.quit, bg='#FF8800', font=('arial', 15, 'bold')).grid(row=12, column=1, sticky=E)
root.mainloop()