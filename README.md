# OCR
Esta é uma API REST, com o intuito de processar imagens, em ordem de reconhecer caracteres, mais especificamente, digitos. 

Utiliza KNN (k-nearest neighbors), efetuando o reconhecimento de digitos usando um conjunto de dados de treinamento diretamente.

# Pre-reqs
- Python3
- Pip

# Instruções de execução

Para executar o projeto:
```
$ pip install -r requirements.txt
$ uvicorn ocr:app --port 8000
```

Para acessar o endpoint de reconhecimento de digitos, as seguinte opções são as mais práticas.

Via shell script:
```
$ test.sh file_name_here.png
```

Via Swagger:
http://127.0.0.1:8000/docs

# To do
- Extração de caracteres
- Reconhecimento de qualquer tipo de caracter