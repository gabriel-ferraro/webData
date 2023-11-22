import os

import matplotlib
import pandas as pd

from flask import Flask, render_template, request, send_file, current_app, redirect, url_for
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import BytesIO
import base64

from src import app
from src.forms import MLForm

matplotlib.use('agg')

@app.route('/', methods=['GET', 'POST'])
def index():
    form = MLForm()

    knn_explanation = (
        'KNN (K-nearest neighbors): é um algoritmo que permite classificar novas amostras a partir da distância em relação às demais amostras do dataset.'
        'Os vizinhos no contexto do KNN são dados existentes no conjunto de treinamento. '
        'O modelo aprende com esses dados e utiliza a proximidade entre novos dados e os dados de treinamento para fazer previsões ou classificações.')
    knn_parameters = ('1o Parametro : N-neighbors (Numero de Vizinhos); \n'
                      '2o Parametro : Weights (Peso - Uniform (Pesos iguais aos vizinhos), Distance (Vizinhos mais próximos tem mais peso)\n'
                      '3o Parâmetro : Leaf Size (Tamanho da folha para os algoritmos ball_tree ou kd_tree) *Minimo 30')

    mlp_explanation = (
        'O MLP (Multilayer Perceptron) é um tipo de modelo de aprendizado de máquina que processa informações em camadas. Ele aprende com dados passados para fazer previsões ou '
        'tomar decisões em novas situações. Cada camada contém "neurônios" que transformam as informações.')
    mlp_parameters = ('1o Parâmetro : Hidden Layer Sizes (Camadas Ocultas)\n'
                      '2o Parâmetro : Max Iter (Número Máximo de Iterações)\n'
                      '3o Parâmetro : Learning Rate (Taxa de Aprendizado) - Deve ser uma string com "invscaling", "constant", "adaptive"')

    dt_explanation = (
        'A Decision Tree (Árvore de Decisão) é um modelo de aprendizado de máquina que toma decisões com base em condicionais. Ela divide os dados em conjuntos menores com base nas '
        'características mais importantes, formando uma estrutura de árvore. Cada divisão é determinada pela característica que melhor separa os dados. Isso continua até que o modelo '
        'crie uma estrutura hierárquica que pode ser usada para fazer previsões ou classificações.')
    dt_parameters = ('1o Parâmetro : Max Depth (Profundidade Máxima)\n'
                     '2o Parâmetro : Random State (Estado Aleatório)\n'
                     '3o Parâmetro : Min Sample Leaf (Numero Minimo de Amostrar em um Nó)')

    rf_explanation = (
        'Random Forest (Floresta Aleatória) é um modelo de aprendizado de máquina que constrói várias árvores de decisão e as combina para fazer previsões mais robustas. Cada árvore é '
        'treinada em uma amostra aleatória dos dados e faz previsões independentes.')
    rf_parameters = ('1o Parâmetro : N Estimators (Número de Árvores);\n'
                     '2o Parâmetro : Max Features (Número máximo de características);\n'
                     '3o Parâmetro : Max Depth (Profundidade Máxima das Árvores)')

    return render_template('index.html', form=form, knn_explanation=knn_explanation, knn_parameters=knn_parameters, mlp_explanation=mlp_explanation, mlp_parameters=mlp_parameters,
                           dt_explanation=dt_explanation, dt_parameters=dt_parameters, rf_explanation=rf_explanation, rf_parameters=rf_parameters)

@app.route('/train', methods=['POST'])
def train():
    classifier_name = request.form.get('classifier')
    parameters = get_parameters(request.form, classifier_name)

    # Carregue o conjunto de dados Iris
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Divida o conjunto de dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inicialize o classificador selecionado com os parâmetros escolhidos
    classifier = get_classifier(classifier_name, parameters)

    # Treine o classificador
    classifier.fit(X_train, y_train)

    # Faça previsões
    y_pred = classifier.predict(X_test)

    # Calcule as métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    # Crie a matriz de confusão
    classes = iris.target_names.tolist()
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes)

    # Converta a imagem para uma string base64
    image_str = plot_to_base64()

    result = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'image': image_str,
        'confusion_matrix_path': 'static/conf_photos/confusion_matrix.png'
    }

    knn_explanation = (
        'KNN (K-nearest neighbors): é um algoritmo que permite classificar novas amostras a partir da distância em relação às demais amostras do dataset.'
        'Os vizinhos no contexto do KNN são dados existentes no conjunto de treinamento. '
        'O modelo aprende com esses dados e utiliza a proximidade entre novos dados e os dados de treinamento para fazer previsões ou classificações.')
    knn_parameters = ('1o Parametro : N-neighbors (Numero de Vizinhos); \n'
                      '2o Parametro : Weights (Peso - Uniform (Pesos iguais aos vizinhos), Distance (Vizinhos mais próximos tem mais peso)\n'
                      '3o Parâmetro : Leaf Size (Tamanho da folha para os algoritmos ball_tree ou kd_tree) *Minimo 30')

    mlp_explanation = (
        'O MLP (Multilayer Perceptron) é um tipo de modelo de aprendizado de máquina que processa informações em camadas. Ele aprende com dados passados para fazer previsões ou '
        'tomar decisões em novas situações. Cada camada contém "neurônios" que transformam as informações.')
    mlp_parameters = ('1o Parâmetro : Hidden Layer Sizes (Camadas Ocultas)\n'
                      '2o Parâmetro : Max Iter (Número Máximo de Iterações)\n'
                      '3o Parâmetro : Learning Rate (Taxa de Aprendizado) - Deve ser uma string com "invscaling", "constant", "adaptive"')

    dt_explanation = (
        'A Decision Tree (Árvore de Decisão) é um modelo de aprendizado de máquina que toma decisões com base em condicionais. Ela divide os dados em conjuntos menores com base nas '
        'características mais importantes, formando uma estrutura de árvore. Cada divisão é determinada pela característica que melhor separa os dados. Isso continua até que o modelo '
        'crie uma estrutura hierárquica que pode ser usada para fazer previsões ou classificações.')
    dt_parameters = ('1o Parâmetro : Max Depth (Profundidade Máxima)\n'
                     '2o Parâmetro : Random State (Estado Aleatório)\n'
                     '3o Parâmetro : Min Sample Leaf (Numero Minimo de Amostrar em um Nó)')

    rf_explanation = (
        'Random Forest (Floresta Aleatória) é um modelo de aprendizado de máquina que constrói várias árvores de decisão e as combina para fazer previsões mais robustas. Cada árvore é '
        'treinada em uma amostra aleatória dos dados e faz previsões independentes.')
    rf_parameters = ('1o Parâmetro : N Estimators (Número de Árvores);\n'
                     '2o Parâmetro : Max Features (Número máximo de características);\n'
                     '3o Parâmetro : Max Depth (Profundidade Máxima das Árvores)')

    return render_template('index.html', form=MLForm(), result=result, knn_explanation=knn_explanation, knn_parameters=knn_parameters, mlp_explanation=mlp_explanation, mlp_parameters=mlp_parameters,
                           dt_explanation=dt_explanation, dt_parameters=dt_parameters, rf_explanation=rf_explanation, rf_parameters=rf_parameters)
def get_parameters(form_data, classifier_name):
    params = {}
    for i in range(1, 4):
        param_key = f'param{i}'
        param_value = form_data.get(param_key)
        params[param_key] = param_value

    # Lógica específica para cada classificador
    if classifier_name == 'knn':
        params['n_neighbors'] = int(params.get('param1'))
        params['weights'] = params.get('param2')
        params['leaf_size'] = int(params.get('param3'))

    elif classifier_name == 'mlp':
        params['hidden_layer_sizes'] = int(params.get('param1'))
        params['max_iter'] = int(params.get('param2'))
        params['learning_rate'] = params.get('param3')

    elif classifier_name == 'dt':
        params['max_depth'] = int(params.get('param1'))
        params['random_state'] = int(params.get('param2'))
        params['min_samples_leaf'] = int(params.get('param3'))

    elif classifier_name == 'rf':
        params['n_estimators'] = int(params.get('param1'))
        params['max_features'] = int(params.get('param2'))
        params['max_depth'] = int(params.get('param3'))

    return params

def get_classifier(name, params):
    if name == 'knn':
        classifier = KNeighborsClassifier(n_neighbors=params['n_neighbors'], weights=params['weights'], leaf_size=params['leaf_size'])
    elif name == 'mlp':
        classifier = MLPClassifier(hidden_layer_sizes=(params['hidden_layer_sizes'],), max_iter=params['max_iter'], learning_rate=params['learning_rate'])
    elif name == 'dt':
        classifier = DecisionTreeClassifier(max_depth=params['max_depth'], random_state=params['random_state'], min_samples_leaf=params['min_samples_leaf'])
    elif name == 'rf':
        classifier = RandomForestClassifier(n_estimators=params['n_estimators'], max_features=params['max_features'], max_depth=params['max_depth'])

    return classifier

def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    # Verifica se a pasta 'src/static/conf_photos' existe, senão é criada
    conf_photos_dir = 'src/static/conf_photos'
    if not os.path.exists(conf_photos_dir):
        os.makedirs(conf_photos_dir)

    # Salve a matriz de confusão no diretório 'src/static/conf_photos' com um nome específico
    plt.savefig(os.path.join(conf_photos_dir, 'confusion_matrix.png'))

def plot_to_base64():
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_str = base64.b64encode(img.read()).decode()
    return img_str

@app.route('/download_confusion_matrix')
def download_confusion_matrix():
    conf_photos_dir = os.path.join(current_app.root_path, 'static', 'conf_photos')
    file_path = os.path.join(conf_photos_dir, 'confusion_matrix.png')
    return send_file(file_path, as_attachment=True, mimetype='image/png')