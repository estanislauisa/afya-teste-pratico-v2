import os
import PyPDF2

from flask import Flask, request, render_template_string, url_for
from dotenv import load_dotenv

# Importações do LangChain / OpenAI (ajuste conforme a sua instalação)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

############################
# Carrega variáveis de ambiente
############################
load_dotenv()

# Define o Flask, apontando para a pasta de arquivos estáticos
app = Flask(__name__, static_folder='static')

############################
# Funções de backend
############################

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
        return text

def create_rag_pipeline(texts):
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    faiss_index = FAISS.from_texts(texts, embeddings)
    retriever = faiss_index.as_retriever()

    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="gpt-4o-mini",  # Ajuste conforme seu modelo
        temperature=0
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
    return chain

def answer_question(pdf_path, question):
    text = extract_text_from_pdf(pdf_path)
    chain = create_rag_pipeline([text])
    result = chain.invoke({"query": question})
    if isinstance(result, dict):
        return result.get("result", "") or result.get("text", "")
    return result

############################
# Caminho do PDF
############################
PDF_PATH = "data/herois_marvel.pdf"

############################
# HTML + CSS inline
############################

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="utf-8">
    <title>Marvel Chat</title>
    <style>
        /* Reset básico */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: Arial, sans-serif;
        }

        body {
            background: #f1f1f1;
            color: #333;
            padding: 20px;
        }

        /* Container principal */
        .container {
            max-width: 600px;
            margin: 0 auto;
            background: #fff;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 3px 8px rgba(0,0,0,0.2);
        }

        /* Logo Marvel */
        .logo {
            display: block;
            margin: 0 auto 20px;
            max-width: 250px; /* Ajuste conforme sua imagem */
        }

        /* Formulário */
        form {
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-bottom: 20px;
        }

        label {
            font-weight: bold;
        }

        input[type="text"] {
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 4px;
            font-size: 1rem;
        }

        button {
            background-color: #ED1D24;
            color: #fff;
            border: none;
            padding: 12px;
            border-radius: 4px;
            font-size: 1rem;
            cursor: pointer;
            transition: background 0.3s;
        }

        button:hover {
            background-color: #c70000;
        }

        hr {
            margin: 20px 0;
        }

        /* Seções de pergunta e resposta */
        .question, .response {
            background: #fafafa;
            border-left: 4px solid #ED1D24;
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 4px;
        }

        .question strong {
            color: #ED1D24;
        }

        .response {
            border-left-color: #0066cc; 
        }

        .response p {
            font-style: italic;
        }

        /* Rodapé (opcional) */
        .footer {
            text-align: center;
            margin-top: 30px;
            font-size: 0.9rem;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Substituindo o título por uma imagem da Marvel -->
        <img src="{{ url_for('static', filename='marvel_logo.png') }}" alt="Marvel Logo" class="logo">

        <form method="POST" action="/">
            <label for="question">Pergunta:</label>
            <input type="text" id="question" name="question" placeholder="Digite sua pergunta..." required>
            <button type="submit">Enviar</button>
        </form>
        
        {% if response %}
            <div class="question">
                <strong>Sua pergunta:</strong>
                <p>{{ question }}</p>
            </div>
            <div class="response">
                <strong>Resposta:</strong>
                <p>{{ response }}</p>
            </div>
        {% endif %}
        
        <hr>
        <div class="footer">
            <p>© 2025 - Marvel Chat</p>
        </div>
    </div>
</body>
</html>
"""

############################
# Rotas Flask
############################

@app.route("/", methods=["GET", "POST"])
def index():
    response = None
    question = None

    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if question:
            response = answer_question(PDF_PATH, question)

    return render_template_string(HTML_TEMPLATE, question=question, response=response)

############################
# Execução
############################
if __name__ == "__main__":
    # Rode: python app.py
    # Acesse em: http://127.0.0.1:5000
    app.run(host="0.0.0.0", port=5000, debug=True)

