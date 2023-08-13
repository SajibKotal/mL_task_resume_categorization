import os
import argparse
import shutil
import pickle
import re
import nltk
import PyPDF2
import docx2txt
import pdfplumber

#Here Load the Model and Vectorizer

with open('model_onvsRF.pkl',"rb") as f:
    model = pickle.load(f)
    # vectorizer = pickle.load(f)
with open('vector.pkl',"rb") as v:
    vectorizer = pickle.load(v)

print(vectorizer)
#Preproces the text
STEMMER = nltk.stem.porter.PorterStemmer()
def preprocess(txt):
    # convert all characters in the string to lower case
    txt = txt.lower()
    # remove non-english characters, punctuation and numbers
    txt = re.sub('[^a-zA-Z]', ' ', txt)
    # tokenize word
    txt = nltk.tokenize.word_tokenize(txt)
    # remove stop words
    txt = [w for w in txt if not w in nltk.corpus.stopwords.words('english')]
    # stemming
    txt = [STEMMER.stem(w) for w in txt]

    return ' '.join(txt)

# This Function work to extract text from PDF files
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text



#For Creating category resume folder to keept the predected resume in categoriwise
def categorize_resumes(input_dir):
    output_dir = os.path.join(input_dir, 'categorized_resumes')
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(input_dir, 'categorized_resumes.csv')

    with open(output_csv, 'w') as csv_file:
        csv_file.write('filename,category\n')

    for filename in os.listdir(input_dir):
        if filename.endswith('.pdf'):
            pdf_fullpath = os.path.join(input_dir, filename)
            text = extract_text_from_pdf(pdf_fullpath)
            cleaned = preprocess(text)
            text_tfidf = vectorizer.transform([cleaned])
            predicted_category = model.predict(text_tfidf)[0]
            output_category_dir = os.path.join(output_dir, predicted_category)
            os.makedirs(output_category_dir, exist_ok=True)
            shutil.move(pdf_fullpath, os.path.join(output_category_dir, filename))

            with open(output_csv, 'a') as csv_file:
                csv_file.write(f"{filename},{predicted_category}\n")

# For Command Line Execution like python script.py Newfolder
# Newfolder is my path/to/dir that contains the test resume
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Categorize Resumes')
    parser.add_argument('input_dir', type=str, help='Path to directory containing resumes')
    args = parser.parse_args()

    categorize_resumes(args.input_dir)

    print('Resumes categorized and CSV file generated.')


