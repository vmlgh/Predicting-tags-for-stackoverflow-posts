from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
import pandas as pd
import collections
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt


from sklearn.model_selection import train_test_split 

stop_words = set(stopwords.words('english'))

#tag_names = ['android', 'csharp', 'cpp', 'html', 'java', 'javascript', 'jquery', 'php', 'python']


main = tkinter.Tk()
main.title("Multi-class Multi-tag Classifier System for StackOverflow Questions")
main.geometry("1300x1200")

global filename
global cls1,cls2,cls3
global cv1,cv2,cv3
global X_train1, X_test1, y_train1, y_test1
global X_train2, X_test2, y_train2, y_test2
global X_train3, X_test3, y_train3, y_test3

def getTagID(name):
    tid = -1
    for i in range(len(tag_names)):
        if tag_names[i] == name:
            tid = i
            break;
    return tid    


def rem_html_tags(question):
    regex = re.compile('<.*?>')
    return re.sub(regex, '', question)

def removePunct(question):
    question = re.sub('\W+',' ', question)
    question = question.strip()
    return question

def uploadTrain():
    global filename
    filename = filedialog.askopenfilename(initialdir="dataset")
    #pathlabel.config(text=filename)
    textarea.delete('1.0', END)
    textarea.insert(END,"Dataset loaded\n");
                    
def preprocess():
    global tag_names
    ques = pd.read_csv(filename,encoding='iso-8859-1')
    count = 0
    size = 0
    data1 = 'Question,Tag\n'
    data2 = 'Question,Tag\n'
    data3 = 'Question,Tag\n'
    
    tag_names = []

    for i in range(len(ques)):
        size = size + 1
        question = ques.get_value(i, 'Body')
        question = rem_html_tags(question)
        question = removePunct(question)
        tag = ques.get_value(i, 'Tags')
        tag = tag.replace("#", "sharp")
        tag = tag.replace("++","pp")
        tag = removePunct(tag)
        tag = tag.split(" ")
        for k in range(len(tag)):
            if k == 0 and len(tag) > 0:
                data1+=question+","+tag[0]+"\n"
            if k == 1 and len(tag) > 1:
                data2+=question+","+tag[1]+"\n"
            if k == 2 and len(tag) > 2:
                data3+=question+","+tag[2]+"\n"
            tag_names.append(str(tag[k]))
    tag_names = list(set(tag_names))
    print(tag_names)
      
            
    if data1 != 'Question,Tag\n':
        f = open("set/set1.csv", "w")
        f.write(data1)
        f.close()

    if data2 != 'Question,Tag\n':
        f = open("set/set2.csv", "w")
        f.write(data2)
        f.close()

    if data3 != 'Question,Tag\n':
        f = open("set/set3.csv", "w")
        f.write(data3)
        f.close()
    textarea.delete('1.0', END)
    textarea.insert(END,"Total processed questions are : "+str(size)+"\n")
    #textarea.insert(END,"All Set files saved inside Set folder\n")

def vector(file):
    X = []
    Y = []
    train = pd.read_csv(file,encoding='iso-8859-1')
    count = 0
    for i in range(len(train)):
        question = train.get_value(i,0,takeable = True)
        tag = train.get_value(i,1,takeable = True)
        question = question.lower()
        arr = question.split(" ")
        msg = ''
        for k in range(len(arr)):
            word = arr[k].strip()
            if len(word) > 2 and word not in stop_words:
                msg+=word+" "
        text = msg.strip()
        X.append(text)
        Y.append(getTagID(tag))
        
    X = np.asarray(X)   
    Y = np.asarray(Y)
    return X,Y

def countVector():
    textarea.delete('1.0', END)
    global cv1,cv2,cv3
    global X_train1, X_test1, y_train1, y_test1
    global X_train2, X_test2, y_train2, y_test2
    global X_train3, X_test3, y_train3, y_test3
    X1,Y1 = vector('set/set1.csv')
    X2,Y2 = vector('set/set2.csv')
    X3,Y3 = vector('set/set3.csv')

    cv1 = CountVectorizer(analyzer='word',stop_words = stop_words, lowercase = True)
    X1 = cv1.fit_transform(X1).toarray()

    cv2 = CountVectorizer(analyzer='word',stop_words = stop_words, lowercase = True)
    X2 = cv2.fit_transform(X2).toarray()

    cv3 = CountVectorizer(analyzer='word',stop_words = stop_words, lowercase = True)
    X3 = cv3.fit_transform(X3).toarray()

    textarea.insert(END, "  After applying count vectorizer: " + "\n")
    plt.imshow(X1, interpolation='nearest')
    plt.gray()

    #textarea.insert(END,"Set 1 total records : "+str(X1.shape)+"\n")
    #textarea.insert(END,"Set 2 total records : "+str(X2.shape)+"\n")
    #textarea.insert(END,"Set 3 total records : "+str(X3.shape)+"\n")

    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, Y1, test_size = 0.2, random_state = 0)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, Y2, test_size = 0.2, random_state = 0)
    X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, Y3, test_size = 0.2, random_state = 0)
 
    
def dataPreparation():
    textarea.delete('1.0', END)
    textarea.insert(END,"Set 1 train records : "+str(X_train1.shape)+"        Set 1 test records : "+str(X_test1.shape)+"\n")
    textarea.insert(END,"Set 2 train records : "+str(X_train2.shape)+"        Set 2 test records : "+str(X_test2.shape)+"\n")
    textarea.insert(END,"Set 3 train records : "+str(X_train3.shape)+"        Set 3 test records : "+str(X_test3.shape)+"\n")

    
def prediction(X_test, cls):
    y_pred = cls.predict(X_test) 
    return y_pred 
    
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred):
    accuracy = accuracy_score(y_test,y_pred)*100
    return accuracy

def trainSVMClassifier():
    textarea.delete('1.0', END)
    global cls_svm1,cls_svm2,cls_svm3
    global acc_svm
    global recall_svm
    global pres_svm
    global f1_svm
    
    acc_svm = []
    pres_svm = []
    recall_svm = []
    f1_svm = []
    
    #cls_svm1 = svm.SVC(probability=True)
    cls_svm1 = svm.LinearSVC(penalty='l1', intercept_scaling=1, dual=False)
    cls_svm1.fit(X_train1, y_train1)
    y_pred1 = cls_svm1.predict(X_test1) 
    acc_svm.append(accuracy_score(y_test1,y_pred1)*100)
    pres_svm.append(precision_score(y_test1, y_pred1, average='macro'))
    recall_svm.append(recall_score(y_test1, y_pred1, average='macro'))
    print(recall_svm[0])
    f1_svm.append(f1_score(y_test1, y_pred1, average='macro'))
    

    #cls2 = svm.SVC(probability=True)
    cls_svm2 = svm.LinearSVC(penalty='l1', intercept_scaling=1, dual=False)
    cls_svm2.fit(X_train2, y_train2)
    y_pred2 = cls_svm2.predict(X_test2) 
    acc_svm.append(accuracy_score(y_test2,y_pred2)*100)
    pres_svm.append(precision_score(y_test2, y_pred2, average='macro'))
    recall_svm.append(recall_score(y_test2, y_pred2, average='macro'))
    #print(recall_svm[1])
    f1_svm.append(f1_score(y_test2, y_pred2, average='macro'))
    
    

    #cls3 = svm.SVC(probability=True)
    cls_svm3 = svm.LinearSVC(penalty='l1', intercept_scaling=1, dual=False)
    cls_svm3.fit(X_train3, y_train3)
    y_pred3 = cls_svm3.predict(X_test3) 
    acc_svm.append(accuracy_score(y_test3,y_pred3)*100)
    pres_svm.append(precision_score(y_test3, y_pred3, average='macro'))
    recall_svm.append(recall_score(y_test3, y_pred3, average='macro'))
    #print(recall_svm[2])
    f1_svm.append(f1_score(y_test3, y_pred3, average='macro'))
    
    textarea.insert(END,"      SVM  Classifier1  Accuracy :  {:.4f}".format(acc_svm[0])+"\n")
    textarea.insert(END,"      SVM  Classifier2  Accuracy :  {:.4f}".format(acc_svm[1])+"\n")
    textarea.insert(END,"      SVM  Classifier3  Accuracy :  {:.4f}".format(acc_svm[2])+"\n")

    
    
    
def trainNBClassifier():
    textarea.delete('1.0', END)
    global cls_nb1,cls_nb2,cls_nb3
    global acc_nb
    global recall_nb
    global pres_nb
    global f1_nb
    
    acc_nb = []
    pres_nb = []
    recall_nb = []
    f1_nb = []
    
    cls_nb1 = MultinomialNB(fit_prior=True, class_prior=None)
    cls_nb1.fit(X_train1, y_train1)
    y_pred1 = cls_nb1.predict(X_test1) 
    acc_nb.append(accuracy_score(y_test1,y_pred1)*100)
    pres_nb.append(precision_score(y_test1, y_pred1, average='macro'))
    recall_nb.append(recall_score(y_test1, y_pred1, average='macro'))
    f1_nb.append(f1_score(y_test1, y_pred1, average='macro'))
    


    cls_nb2 = MultinomialNB(fit_prior=True, class_prior=None)
    cls_nb2.fit(X_train2, y_train2)
    y_pred2 = cls_nb2.predict(X_test2) 
    acc_nb.append(accuracy_score(y_test2,y_pred2)*100)
    pres_nb.append(precision_score(y_test2, y_pred2, average='macro'))
    recall_nb.append(recall_score(y_test2, y_pred2, average='macro'))
    f1_nb.append(f1_score(y_test2, y_pred2, average='macro'))
    
    

    cls_nb3 = MultinomialNB(fit_prior=True, class_prior=None)
    cls_nb3.fit(X_train3, y_train3)
    y_pred3 = cls_nb3.predict(X_test3) 
    acc_nb.append(accuracy_score(y_test3,y_pred3)*100)
    pres_nb.append(precision_score(y_test3, y_pred3, average='macro'))
    recall_nb.append(recall_score(y_test3, y_pred3, average='macro'))
    f1_nb.append(f1_score(y_test3, y_pred3, average='macro'))
    
    textarea.insert(END,"      NB  Classifier1  Accuracy :  {:.4f}".format(acc_nb[0])+"\n")
    textarea.insert(END,"      NB  Classifier2  Accuracy :  {:.4f}".format(acc_nb[1])+"\n")
    textarea.insert(END,"      NB  Classifier3  Accuracy :  {:.4f}".format(acc_nb[2])+"\n")

    
    
def trainLRClassifier():
    textarea.delete('1.0', END)
    global cls_lr1,cls_lr2,cls_lr3
    global acc_lr
    global recall_lr
    global pres_lr
    global f1_lr
    
    acc_lr = []
    pres_lr = []
    recall_lr = []
    f1_lr = []
    
    cls_lr1 = LogisticRegression()
    cls_lr1.fit(X_train1, y_train1)
    y_pred1 = cls_lr1.predict(X_test1) 
    acc_lr.append(accuracy_score(y_test1,y_pred1)*100)
    pres_lr.append(precision_score(y_test1, y_pred1, average='macro'))
    recall_lr.append(recall_score(y_test1, y_pred1, average='macro'))
    f1_lr.append(f1_score(y_test1, y_pred1, average='macro'))
    


    cls_lr2 = LogisticRegression()
    cls_lr2.fit(X_train2, y_train2)
    y_pred2 = cls_lr2.predict(X_test2) 
    acc_lr.append(accuracy_score(y_test2,y_pred2)*100)
    pres_lr.append(precision_score(y_test2, y_pred2, average='macro'))
    recall_lr.append(recall_score(y_test2, y_pred2, average='macro'))
    f1_lr.append(f1_score(y_test2, y_pred2, average='macro'))
    
    

    cls_lr3 = LogisticRegression()
    cls_lr3.fit(X_train3, y_train3)
    y_pred3 = cls_lr3.predict(X_test3) 
    acc_lr.append(accuracy_score(y_test3,y_pred3)*100)
    pres_lr.append(precision_score(y_test3, y_pred3, average='macro'))
    recall_lr.append(recall_score(y_test3, y_pred3, average='macro'))
    f1_lr.append(f1_score(y_test3, y_pred3, average='macro'))
    
    
    
    textarea.insert(END,"      LR  Classifier1  Accuracy :  {:.4f}".format(acc_lr[0])+"\n")
    textarea.insert(END,"      LR  Classifier2  Accuracy :  {:.4f}".format(acc_lr[1])+"\n")
    textarea.insert(END,"      LR  Classifier3  Accuracy :  {:.4f}".format(acc_lr[2])+"\n")

    

def processLine(line):
    line = rem_html_tags(line)
    line = removePunct(line)
    msg = ''
    arr = line.split(' ')
    for i in range(len(arr)):
        arr[i] = arr[i].strip()
        if len(arr[i]) > 2 and arr[i] not in stop_words:
            msg+=arr[i]+" "
    msg = msg.strip();
    print(msg)
    return msg

def predict():
    textarea.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="dataset")
    index = 1
    with open(filename, "r") as file: #reading emotion word
        for line in file:
            ques = line
            line = line.strip('\n')
            line = line.strip()
            temp = line
            line = line.lower()
            line = processLine(line)
            cv = CountVectorizer(vocabulary=cv1.get_feature_names(),stop_words = "english", lowercase = True)
            test1 = cv.fit_transform([line])
            cv = CountVectorizer(vocabulary=cv2.get_feature_names(),stop_words = "english", lowercase = True)
            test2 = cv.fit_transform([line])
            cv = CountVectorizer(vocabulary=cv3.get_feature_names(),stop_words = "english", lowercase = True)
            test3 = cv.fit_transform([line])


            c1 = cls_lr1.predict(test1.toarray())[0]
            c2 = cls_lr2.predict(test2.toarray())[0]
            c3 = cls_lr3.predict(test3.toarray())[0]
            
            final_tags = [tag_names[c1], tag_names[c2], tag_names[c3]]
            final_tags = list(dict.fromkeys(final_tags))
            textarea.insert(END, 'Question- '+ques+ "\n TAG Predicted AS : ")
            textarea.insert(END, str(final_tags)+"\n\n")
            index = index + 1
    
def Pres_graph():
    
    N = 3
    ind = np.arange(N)  # the x locations for the groups
    width = 0.27       # the width of the bars

    fig = plt.figure()
    ax = fig.add_subplot(111)

    rects1 = ax.bar(ind, pres_svm, width, color='indianred')
    rects2 = ax.bar(ind+width, pres_nb, width, color='goldenrod')
    rects3 = ax.bar(ind+width*2, pres_lr, width, color='lightseagreen')

    ax.set_ylabel('Precision Values')
    ax.set_xticks(ind+width)
    ax.set_xticklabels( ('Classifer1', 'Classifer2', 'Classifer3') )
    ax.legend( (rects1[0], rects2[0], rects3[0]), ('SVM', 'NB', 'LR') )

    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.text(rect.get_x() - 0.05, 1.02*h, "{:.2}".format(h))

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    plt.show()
    
def Rec_graph():

    N = 3
    ind = np.arange(N)  # the x locations for the groups
    width = 0.27       # the width of the bars

    fig = plt.figure()
    ax = fig.add_subplot(111)

    rects1 = ax.bar(ind, recall_svm, width, color='indianred')
    rects2 = ax.bar(ind+width, recall_nb, width, color='goldenrod')
    rects3 = ax.bar(ind+width*2, recall_lr, width, color='lightseagreen')

    ax.set_ylabel('Recall Values')
    ax.set_xticks(ind+width)
    ax.set_xticklabels( ('Classifer1', 'Classifer2', 'Classifer3') )
    ax.legend( (rects1[0], rects2[0], rects3[0]), ('SVM', 'NB', 'LR') )

    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.text(rect.get_x() - 0.05, 1.02*h, "{:.2}".format(h))

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    plt.show()
    
def F1_graph():
    
    N = 3
    ind = np.arange(N)  # the x locations for the groups
    width = 0.27       # the width of the bars

    fig = plt.figure()
    ax = fig.add_subplot(111)

    rects1 = ax.bar(ind, f1_svm, width, color='indianred')
    rects2 = ax.bar(ind+width, f1_nb, width, color='goldenrod')
    rects3 = ax.bar(ind+width*2, f1_lr, width, color='lightseagreen')

    ax.set_ylabel('F1 Scores')
    ax.set_xticks(ind+width)
    ax.set_xticklabels( ('Classifer1', 'Classifer2', 'Classifer3') )
    ax.legend( (rects1[0], rects2[0], rects3[0]), ('SVM', 'NB', 'LR') )

    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.text(rect.get_x() - 0.05, 1.02*h, "{:.2}".format(h))

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    plt.show()

def close():
    global main
    main.destroy()
    
font = ('times', 16, 'bold')
title = Label(main, text='Multi-class Multi-tag Classifier System for StackOverflow Questions')
title.config(bg='mint cream', fg='olive drab')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
uploadButton = Button(main, text="Upload Stackoverflow Dataset", command=uploadTrain)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

#pathlabel = Label(main)
#pathlabel.config(bg='mint cream', fg='olive drab')  
#pathlabel.config(font=font1)           
#pathlabel.place(x=460,y=100)

processButton = Button(main, text="Preprocess Questions", command=preprocess)
processButton.place(x=450,y=100)
processButton.config(font=font1) 

countButton = Button(main, text="Count Vectorization", command=countVector)
countButton.place(x=700,y=100)
countButton.config(font=font1) 

dataButton = Button(main, text="Data Preparation For Classifier", command=dataPreparation)
dataButton.place(x=950,y=100)
dataButton.config(font=font1)

trainButton = Button(main, text="Train SVM Classifier", command=trainSVMClassifier)
trainButton.place(x=100,y=170)
trainButton.config(font=font1) 

trainButton = Button(main, text="Train NB Classifier", command=trainNBClassifier)
trainButton.place(x=330,y=170)
trainButton.config(font=font1) 

trainButton = Button(main, text="Train LR Classifier", command=trainLRClassifier)
trainButton.place(x=560,y=170)
trainButton.config(font=font1) 

predictButton = Button(main, text="Upload Test data", command=predict)
predictButton.place(x=890,y=170)
predictButton.config(font=font1) 

graphButton = Button(main, text="Precision Graph", command=Pres_graph)
graphButton.place(x=350,y=240)
graphButton.config(font=font1)

graphButton = Button(main, text="Recall Graph", command=Rec_graph)
graphButton.place(x=580,y=240)
graphButton.config(font=font1)

graphButton = Button(main, text="F1-score Graph", command=F1_graph)
graphButton.place(x=810,y=240)
graphButton.config(font=font1)

font1 = ('times', 12, 'bold')
textarea=Text(main,height=20,width=158)
scroll=Scrollbar(textarea)
textarea.configure(yscrollcommand=scroll.set)
textarea.place(x=20,y=300)
textarea.config(font=font1)


main.config(bg='gainsboro')
main.mainloop()
