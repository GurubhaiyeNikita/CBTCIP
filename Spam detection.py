import warnings
warnings.filterwarnings("ignore")        #to avoid any warning shown on the output screen

import string
import pandas as p                      #for data access,manipulation

import matplotlib.pyplot as pt          #for visualizations
import matplotlib as mt
import seaborn as sb                    #advance visualization tools heatmap

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

#for the  classifiers
from sklearn.linear_model import LogisticRegression       #supervised ML
from sklearn.svm import SVC                               #supervised ML
from sklearn.naive_bayes import MultinomialNB              
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


import nltk                              #for text processing getting stopwords from the message
nltk.download("stopwords")

file=p.read_excel("Spam Email Detection.xlsx")
#df=p.read_csv("Spam Email Detection.csv")

file.head(10)
file.tail(5)

file.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
file

file.rename(columns={'v1': 'label', 'v2':'msg'})
file

file.describe()
file.groupby('label').describe()
file.groupby('msg').describe()

file.shape
file.corr

file['length']=file['msg'].astype(str).apply(len)

txt=file['msg'].copy()


def txt_process(text):
    text=str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))  #to translate text  in  words '
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    return " ".join(text)

txt=txt.apply(txt_process())


vectorizer = TfidfVectorizer(max_features=3000)         #for the text trainig samples tfidvectoriser is used
features = vectorizer.fit_transform(text_feat)

features_train, features_test, labels_train, labels_test = train_test_split(features, file['label'], test_size=0.3, random_state=111)
#splitted the dataset into training and test samples


vc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier(n_neighbors=50)
mnb = MultinomialNB(alpha=0.2)
dtc = DecisionTreeClassifier(min_samples_split=5, random_state=115)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=30, random_state=115)
abc = AdaBoostClassifier(n_estimators=60, random_state=115)
bc = BaggingClassifier(n_estimators=9, random_state=115)
etc = ExtraTreesClassifier(n_estimators=9, random_state=115)

classifier = {
    'SVC' : svc,
    'KN' : knc, 
    'NB': mnb, 
    'DT': dtc, 
    'LR': lrc, 
    'RF': rfc, 
    'AdaBoost': abc,
    'BgC': bc,
    'ETC': etc
    }

def train_classifier(classifier, feature_train, labels_train):    #to iteratively call the function
    classifier.fit(feature_train, labels_train)


def predict_labels(clf, features):                              #to call function and store the resultin iteration
    return (classifier.predict(features))


                                        #for data visualisation

#for storing mettric values of each classifier
accuracy=[]
precision=[]
recall=[]
f1=[]
metric=[]
clf=[]

#to find max  values in the metric
m_acc=0
m_pre=0
m_rc=0
m_f=0


#for training models
for keys,val in classifier.items():
    train_classifier(val, features_train, labels_train)
    predicted = predict_labels(val,features_test)

    a=accuracy_score(labels_test,predicted)
    metric.append((keys, [accuracy_score(labels_test,predicted)]))

    b=precision_score(labels_test,predicted,pos_label="spam")
    precision.append(bytes)

    r=recall_score(labels_test,predicted,pos_label="spam",average="binary")
    recall.append(r)


    f=f1_score(labels_test,predicted,pos_label="spam")
    f1.append(f)
    clf.append(keys)
    
    if m_acc<a:
        m_acc=a
        acc_model=keys

    if m_pre<b:
        m_pre=b
        pre_model=keys

    if m_rc<r:
        m_rc=r
        rc_model=keys
 
    if m_f<f:
        m_f=f
        f_model=keys


#to show maximum values of each metric with model name
print("\nThe highest accuracy score amongst these models is ",m_acc,'with ',acc_model,'model')
print("\nThe highest precision score amongst these models is ",m_pre,'with ',pre_model,'model')
print("\nThe highest recall score amongst these models is ",m_rc,'with ',rc_model,'model')
print("\nThe highest F1 score amongst these models is ",m_f,'with ',f_model,'model')


#dataframe for plotting bar graph
l=list(zip(clf,accuracy,precision,recall,f1))
df=p.DataFrame(l,columns=['Model name','Accuracy','Precision','Recall score','F1 score'])
#print(df)

df.plot(kind="bar",ylim=(0.8,1.0), figsize=(11,6), align='center', colormap="Accent")
pt.xticks(df.index,c)
pt.ylabel('Scores')
pt.title('Distribution by Classifier')
pt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)



#dataframe for plotting heatmap
df1=list(zip(accuracy,precision,recall,f1))
label=['Accuracy','Precision','Recall score','F1 score']


#heatmap for metric comparison
pt.figure(figsize=(6, 3))
sb.heatmap(df1, annot=True, fmt=".2f", cbar=False, cmap="summer_r", xticklabels=label, yticklabels=clf)
pt.title("Metric Comparison")
pt.yticks(rotation=0)
pt.xlabel("Metrics")
pt.ylabel("Models")
pt.tight_layout()
pt.show()


#confusion matrix using heatmap 
k=0
fig, axes = pt.subplots(3, 3, figsize=(15, 10))
for i,g in (classifier.items()):
    prediction = g.predicted(features_test)
    cm = confusion_matrix(labels_test, prediction)
    model_name=i
    row=k//3 
    col=k%3  
    sb.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[row, col]) 
    axes[row, col].set_title(f"{model_name} - Confusion Matrix") 
    axes[row, col].set_xlabel("Predicted") 
    axes[row, col].set_ylabel("Actual") 
    k=k+1
pt.tight_layout()
pt.show()


#final  message to be classified
print("we will classify the sample mail with Multinomial Naive Bayes model used for")
email=input("\n\nenter mail to be classified")

mm=mnb.predict(email)                               #using Multinomial NB ttechnique to classify
if mm[1]==0:                                #as it returned array[0,1] indicating 2 classes ['ham','spam']
  print("\nits not spam")
else:
  print("\nits spam message")


    







