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

X_data = pickle.load(open('D:\Workspace\DoAN\SourceCode\TopicNews\X_data.pkl', 'rb'))
y_data = pickle.load(open('D:\Workspace\DoAN\SourceCode\TopicNews\y_data.pkl', 'rb'))

encoder = preprocessing.LabelEncoder()
y_data_n = encoder.fit_transform(y_data)
encoder.classes_

modelSVM = svm.SVC()

modelNaive = naive_bayes.MultinomialNB()

def train_model_NB( X_data, y_data):       
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.5, random_state=42)
    
    modelNaive.fit(X_train, y_train)
    
    val_predictions = modelNaive.predict(X_test)
    
    print("Validation accuracy: ", metrics.accuracy_score(val_predictions, y_test))


def train_model_SVM( X_data, y_data):       
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.5, random_state=42)
    
    modelSVM.fit(X_train, y_train)
    
    val_predictions = modelSVM.predict(X_test)

    print("Validation accuracy: ", metrics.accuracy_score(val_predictions, y_test))


count_vect = CountVectorizer(
    analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(X_data)

X_data_count = count_vect.transform(X_data)

train_model_NB(X_data_count, y_data)
print("====SVM")
train_model_SVM(X_data_count, y_data)

def preprocessing_doc(doc):
    lines = gensim.utils.simple_preprocess(doc)
    lines = ' '.join(lines)
    lines = ViTokenizer.tokenize(lines)

    return lines

test_doc = '''Thành viên Thaibev vừa chi tiền để nắm 52,91% cổ phần sau đợt tăng gấp đôi vốn điều lệ của Công ty cổ phần Bia Sài Gòn - Lâm Đồng.

Thai Beverage Pubilc Co. Ltd, thành viên của tập đoàn Thaibev, vừa thông báo công ty con gián tiếp là Bia Sài Gòn – Lâm Đồng đã tăng gấp đôi vốn điều lệ lên 200 tỷ đồng vào đầu tuần này. 

Đơn vị trực tiếp sở hữu Bia Sài Gòn – Lâm Đồng là Tổng công ty Bia – Rượu – Nước giải khát Sài Gòn (Sabeco). Tại thời điểm doanh nghiệp này thành lập, Sabeco chỉ nắm 20% vốn. Kế hoạch tăng vốn để mở rộng công suất hoạt động được doanh nghiệp dẫn đầu thị phần bia nội địa thông qua vào đầu tháng 5.

Sabeco góp thêm khoảng 86 tỷ đồng trong đợt tăng vốn mới, qua đó nâng tỷ lệ sở hữu lên 52,91%. Hai cổ đông sáng lập khác nắm chưa đến 2% sau khi tăng vốn. Phần còn lại thuộc về nhóm cổ đông khác có trụ sở tại Lâm Đồng.

Bia Sài Gòn - Lâm Đồng cũng thay đổi Chủ tịch HĐQT, từ ông Nguyễn Thành Nam (nguyên Tổng giám đốc Sabeco) thành ông Teo Hong Keng (Phó Tổng giám đốc Sabeco hiện tại).

Công ty của tỷ phú Charoen Sirivadhanabhakdi cho biết, Sabeco sử dụng dòng tiền từ hoạt động để tài trợ cho việc tăng vốn, qua đó giúp Thaibev có thêm một công ty con hoạt động trong lĩnh vực sản xuất bia rượu tại Việt Nam. Việc góp vốn này không tác động trọng yếu đến thu nhập trên mỗi cổ phiếu và tài sản ròng của phía Thaibev.
'''

test_doc = preprocessing_doc(test_doc)
test_doc_count = count_vect.transform([test_doc])
# print(modelSVM.predict(test_doc_count))
# print(modelNaive.predict(test_doc_count))

# pickle.dump(modelNaive, open('./TopicNews/modelNaiveVi', 'wb'))
# pickle.dump(modelSVM, open('./TopicNews/modelSVMVi', 'wb'))

loaded_modelNaive = pickle.load(open('./TopicNews/modelNaiveVi', 'rb'))
loaded_modelSVM = pickle.load(open('./TopicNews/modelSVMVi', 'rb'))
resultNaive = loaded_modelNaive.predict(test_doc_count)
resultSVM = loaded_modelSVM.predict(test_doc_count)
print(resultNaive)
print(resultSVM)

