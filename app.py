from flask import Flask, render_template, request
import pickle
import gensim
from pyvi import ViTokenizer
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# modelNaive = pickle.load(open('modelNaiveEn','rb'))
# modelSVM = pickle.load(open('modelSVM', 'rb'))

x_data_en = pickle.load(open('X_data_En.pkl', 'rb'))
count_vect_en = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect_en.fit(x_data_en)

x_data_vi = pickle.load(open('X_data_Vi.pkl', 'rb'))
count_vect_vi = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect_vi.fit(x_data_vi)

def preprocessing_doc(doc):
    lines = gensim.utils.simple_preprocess(doc)
    lines = ' '.join(lines)
    # lines = ViTokenizer.tokenize(lines)

    return lines 

def preprocessing_doc_vi(doc):
    lines = gensim.utils.simple_preprocess(doc)
    lines = ' '.join(lines)
    lines = ViTokenizer.tokenize(lines)

    return lines


@app.route('/')
def renderIndex():
    return render_template('index.html')

@app.route('/en/svm')
def renderEnSVMTemp():
    return render_template('en_svm.html')

@app.route('/en/naive')
def renderEnNaiveTemp():
    return render_template('en_naive.html')

@app.route('/vi/svm')
def renderViNaiveTemp():
    return render_template('vi_svm.html')

@app.route('/vi/naive')
def renderViSVMTemp():
    return render_template('vi_naive.html')

@app.route("/en/naive", methods=['POST'])
def printResultNB():
    content_in_form = request.form['content']
    content_in_form_r = content_in_form
    content_in_form = preprocessing_doc(content_in_form)
    test_doc_count = count_vect_en.transform([content_in_form])
    final_result = pickle.load(open('modelNaiveEn', 'rb')).predict(test_doc_count)
    print(final_result)      
    return render_template('en_naive.html', r=final_result, c = content_in_form_r)

@app.route("/en/svm", methods=['POST'])
def printResultSVM():
    content_in_form = request.form['content']
    content_in_form_r = content_in_form
    content_in_form = preprocessing_doc(content_in_form)
    test_doc_count = count_vect_en.transform([content_in_form])
    final_result = pickle.load(open('modelSVMEn', 'rb')).predict(test_doc_count)
    print(final_result)      
    return render_template('en_svm.html', r=final_result, c = content_in_form_r)

@app.route("/vi/svm", methods=['POST'])
def printResultSVMVi():
    content_in_form = request.form['content']
    content_in_form_r = content_in_form
    content_in_form = preprocessing_doc_vi(content_in_form)
    test_doc_count = count_vect_vi.transform([content_in_form])
    final_result = pickle.load(open('modelSVMVi', 'rb')).predict(test_doc_count)   
    return render_template('vi_svm.html', r=final_result, c = content_in_form_r)

@app.route("/vi/naive", methods=['POST'])
def printResultNaiveVi():
    content_in_form = request.form['content']
    content_in_form_r = content_in_form
    content_in_form = preprocessing_doc_vi(content_in_form)
    test_doc_count = count_vect_vi.transform([content_in_form])
    final_result = pickle.load(open('modelNaiveVi', 'rb')).predict(test_doc_count)    
    return render_template('vi_naive.html', r=final_result, c = content_in_form_r)

if __name__ == '__main__':
    app.run()
