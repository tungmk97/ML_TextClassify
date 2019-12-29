from pyvi import ViTokenizer, ViPosTagger # thư viện NLP tiếng Việt
from tqdm import tqdm
import numpy as np
import gensim # thư viện NLP
import pickle
from sklearn.model_selection import train_test_split
import os 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import gensim
from pyvi import ViTokenizer, ViPosTagger
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.svm import LinearSVC

# load data from pkl
X_data = pickle.load(open('D:\Workspace\DoAN\SourceCode\BBCTopic\X_data.pkl', 'rb'))
y_data = pickle.load(open('D:\Workspace\DoAN\SourceCode\BBCTopic\y_data.pkl', 'rb'))

encoder_label = preprocessing.LabelEncoder()
y_data_n = encoder_label.fit_transform(y_data)
encoder_label.classes_

modelSVM = svm.SVC()

modelNaive = naive_bayes.MultinomialNB()
def train_model_naivebayes(X_data, y_data):       
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42)
    modelNaive.fit(X_train, y_train)
    val_predictions = modelNaive.predict(X_test)
    
    print("Validation accuracy: ", metrics.accuracy_score(
                                val_predictions, y_test))
 

def train_model_SVM(X_data, y_data):       
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42)
    # modelSVM = svm.SVC()
    modelSVM.fit(X_train, y_train)
    val_predictions = modelSVM.predict(X_test)

    print("Validation accuracy: ", metrics.accuracy_score(
                                val_predictions, y_test))

count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(X_data)
X_data_count = count_vect.transform(X_data)
# transform the training and validation data using count vectorizer object


#train naive_bayes


# train_model_naivebayes(X_data_count, y_data)
# train_model_SVM(X_data_count, y_data)

def preprocessing_doc(doc):
    lines = gensim.utils.simple_preprocess(doc)
    lines = ' '.join(lines)
    # lines = ViTokenizer.tokenize(lines)

    return lines

test_doc = '''The Competition and Markets Authority said in a statement Friday that it had opened a "phase 2" probe after the companies failed to address its concerns about how the deal would affect the market for online deliveries of restaurant meals and groceries.
        "There's a real risk that it could leave customers, restaurants and grocers facing higher prices and lower quality services as these markets develop," Andrea Gomes da Silva, the regulator's executive director, said in statement earlier this month.
The in-depth investigation could delay the completion of the deal, hurting Deliveroo in the ultracompetitive UK market while giving rivals Just Eat (JSTTY) and Uber (UBER) Eats a boost.
        A spokesperson for Deliveroo said the company would continue to work closely with the regulator.
"We are confident that we will persuade the [Competition and Markets Authority] of the facts that this minority investment will add to competition, helping restaurants to grow their businesses, creating more work for riders, and increasing choice for customers," the spokesperson said in a statement
'''


test_doc = preprocessing_doc(test_doc)
print(test_doc)
test_doc_count = count_vect.transform([test_doc])
print(test_doc_count)
# print(modelNaive.predict(test_doc_count))
# print(modelSVM.predict(test_doc_count))

# pickle.dump(modelNaive, open('./BBCTopic/modelNaiveEn', 'wb'))
# pickle.dump(modelSVM, open('./BBCTopic/modeSVMEn', 'wb'))


loaded_modelNaive = pickle.load(open('./BBCTopic/modelNaiveEn', 'rb'))
loaded_modelSVM = pickle.load(open('./BBCTopic/modelSVMEn', 'rb'))
resultNaive = loaded_modelNaive.predict(test_doc_count)
resultSVM = loaded_modelSVM.predict(test_doc_count)
print(resultNaive)
print(resultSVM)


