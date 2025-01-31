import os
import re
import glob
import base64
import numpy as np
import pandas as pd
import torch
import pdfplumber
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from io import BytesIO
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
from django.shortcuts import render
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer, BertModel

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # Tokenizer for BERT model
model = BertModel.from_pretrained('bert-base-uncased')  # BERT model for text embeddings

nltk.download('punkt')  # Tokenization
nltk.download('punkt_tab')
nltk.download('stopwords')  # Stopwords
nltk.download('wordnet')  # Lemmatization
nltk.download('omw-1.4')

# Creating important objects
stop_words = set(stopwords.words('english'))  # Set of English stopwords
stemmer = PorterStemmer()  # Stemmer for word stemming
lemmatizer = WordNetLemmatizer() 

def p4_upload_plagi(request):
    if request.method == 'POST':
        if 'file' not in request.FILES:
            return JsonResponse({'message': 'No files uploaded'}, status=400)

        files = request.FILES.getlist('file')
        if not files:
            return JsonResponse({'message': 'No files selected'}, status=400)

        uploaded_files = []
        num_files_uploaded = 0
        invalid_files = []
        filenames = []

        upload_dir = settings.P4_PDF_FILES_UPLOAD_DIR
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
            print(f'Created upload directory: {upload_dir}')

        fs = FileSystemStorage(location=upload_dir)

        for file in files:
            if file.name.endswith('.pdf'):
                filename = fs.save(file.name, file)
                filenames.append(filename)
                file_path = os.path.join(upload_dir, filename)
                num_files_uploaded += 1
                uploaded_files.append(file_path)
                print(f'File uploaded: {file_path}')
            else:
                invalid_files.append(file.name)

        if invalid_files:
            return JsonResponse({
                'message': 'Invalid file formats uploaded (only PDF(s) allowed)',
                'invalid_files': invalid_files
            }, status=400)

        if num_files_uploaded < 2:
            return JsonResponse({'message': 'Upload at least two PDF files to check plagiarism...'}, status=400)

        return JsonResponse({'message': f'Successfully uploaded {num_files_uploaded} PDF files', 'uploaded_files': uploaded_files})

    return JsonResponse({'message': 'Invalid request method'}, status=405)

def p4_text_to_vector_bert(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def p4_convert_texts_to_vectors_bert(texts, tokenizer, model):
    vectors = [p4_text_to_vector_bert(text, tokenizer, model) for text in texts]
    return np.array(vectors)

def p4_preprocess_texts(texts):
    preprocessed_texts = []
    for text in texts:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words]
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        preprocessed_texts.append(' '.join(tokens))
    return preprocessed_texts

def p4_extract_text_from_pdf(file):
    pdf_text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            pdf_text += page.extract_text() or ""
    return pdf_text

def p4_textInHandle(request):
    if request.method == 'POST':
        text1 = request.POST.get('text1')
        text2 = request.POST.get('text2')
        texts = [text1, text2]
        preprocessed_texts = p4_preprocess_texts(texts)
        vectors = p4_convert_texts_to_vectors_bert(preprocessed_texts, tokenizer, model)
        distances = np.zeros((2, 2))
        results = []
        for i in range(2):
            for j in range(2):
                if i != j:
                    distances[i, j] = np.sum(np.abs(vectors[i] - vectors[j]))
        for i in range(2):
            for j in range(i + 1, 2):
                distance = distances[i, j]
                is_plagiarized = distance < 50
                results.append(f"Distance: {round(distance, 2)}")
                results.append(is_plagiarized)
        distance_df = pd.DataFrame(distances, index=['Text 1', 'Text 2'], columns=['Text 1', 'Text 2'])
        plt.figure(figsize=(4, 4), dpi=80)
        sns.heatmap(distance_df, annot=True, cmap='coolwarm', fmt='.2f', annot_kws={"size": 12, "color": "white"})
        plt.title('Manhattan Distance Matrix', color='white')
        plt.xlabel('Texts', color='white')
        plt.ylabel('Texts', color='white')
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()
        buffer.seek(0)
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        response_data = {'result': results, 'heatmap': img_str}
        return render(request, 'p4_textResult.html', response_data)
    return JsonResponse({'error': 'An error occurred'})

def p4_detect_plagi(request):
    try:
        texts = []
        files_folder = '/srv/portfolio_project/media/plagi_files/*.pdf'
        pdf_files = glob.glob(files_folder)
        if not pdf_files:
            return JsonResponse({'message': 'No uploaded files found.'}, status=400)
        fnames = [os.path.basename(pdf) for pdf in pdf_files]
        for pdf_path in pdf_files:
            with open(pdf_path, 'rb') as file:
                text = p4_extract_text_from_pdf(file)
                texts.append(text)
        preprocessed_texts = p4_preprocess_texts(texts)
        vectors = p4_convert_texts_to_vectors_bert(preprocessed_texts, tokenizer, model)
        num_vectors = vectors.shape[0]
        distances = np.zeros((num_vectors, num_vectors))
        results = []
        for i in range(num_vectors):
            for j in range(num_vectors):
                if i != j:
                    distances[i, j] = np.sum(np.abs(vectors[i] - vectors[j]))
        for i in range(num_vectors):
            for j in range(i + 1, num_vectors):
                distance = distances[i, j]
                is_plagiarized = distance < 50
                results.append((f"{fnames[i]} - {fnames[j]}", round(distance, 2), str(is_plagiarized)))
        distance_df = pd.DataFrame(distances, index=fnames, columns=fnames)
        plt.figure(figsize=(7, 4), dpi=80)
        sns.heatmap(distance_df, annot=True, cmap='coolwarm', fmt='.2f', annot_kws={"size": 8})
        plt.title('Manhattan Distance Matrix')
        plt.xlabel('Document Names')
        plt.ylabel('Document Names')
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()
        buffer.seek(0)
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        response_data = {'text_result': results, 'visualize': img_str}
        return render(request, 'p4result.html', response_data)
    except Exception as e:
        return JsonResponse({'message': str(e)}, status=400)