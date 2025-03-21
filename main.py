import os
import PyPDF2
import threading

from dotenv import load_dotenv
from PIL import Image

# Agora importamos do langchain_openai, conforme recomendação:
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

import customtkinter as ctk

############################
# Carrega variáveis de ambiente
############################
load_dotenv()

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
    # Cria embeddings usando OpenAIEmbeddings do langchain_openai
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    faiss_index = FAISS.from_texts(texts, embeddings)
    retriever = faiss_index.as_retriever()

    # Usa ChatOpenAI do langchain_openai
    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="gpt-4o-mini",  # Ajuste para seu modelo
        temperature=0
    )
    # Cria a RetrievalQA
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
    return chain

def answer_question(pdf_path, question):
    text = extract_text_from_pdf(pdf_path)
    chain = create_rag_pipeline([text])
    # Em vez de chain.run(question), use chain.invoke(...)
    result = chain.invoke({"query": question})
    # No caso do "stuff", possivelmente result já é string.
    # Se for um dicionário, faça result["result"] ou result["text"] etc.
    # Teste para verificar qual a saída exata no seu caso.
    return result

############################
# Funções de UI e Thread
############################

def add_message(role, text, bubble_color=None):
    """
    Cria uma 'bolha' no chat_frame e retorna o FRAME criado
    (para permitir .destroy() ou atualizações).
    """
    if role == "Você":
        bg = bubble_color if bubble_color else "#444444"  # cinza escuro p/ usuário
        anchor_side = "e"
    else:
        bg = bubble_color if bubble_color else "#ED1D24"  # vermelho p/ assistente
        anchor_side = "w"

    msg_frame = ctk.CTkFrame(
        chat_frame,
        corner_radius=12,
        fg_color=bg,
        width=320
    )
    msg_frame.pack(pady=5, padx=10, anchor=anchor_side)

    name_anchor = "e" if role == "Você" else "w"
    name_label = ctk.CTkLabel(
        msg_frame,
        text=role,
        text_color="#fff",
        font=("Arial", 10, "bold")
    )
    name_label.pack(anchor=name_anchor, padx=10, pady=(5, 0))

    msg_label = ctk.CTkLabel(
        msg_frame,
        text=text,
        wraplength=300,
        justify="left",
        text_color="#fff",
        font=("Arial", 11)
    )
    msg_label.pack(anchor=name_anchor, padx=10, pady=(0,5))

    chat_frame._parent_canvas.yview_moveto(1.0)
    return msg_frame

def process_question(question, placeholder_frame):
    """
    Roda em outra thread p/ não travar a GUI.
    """
    response = answer_question(pdf_path, question)
    # Se "response" for um dicionário, adapte (ex: response["result"] ou algo do tipo)
    if isinstance(response, dict):
        response = response.get("result", "") or response.get("text", "")

    # Finaliza na main thread
    app.after(0, lambda: finalize_assistant_message(placeholder_frame, response))

def finalize_assistant_message(placeholder_frame, response):
    """
    Remove a bolha "Digitando..." e adiciona a bolha final.
    """
    placeholder_frame.destroy()
    add_message("Assistente", response)

def send_question(event=None):
    question = question_var.get().strip()
    if not question:
        return

    # Cria bolha do usuário
    add_message("Você", question)
    question_var.set("")

    # Bolha placeholder do assistente
    placeholder_frame = add_message("Assistente", "Digitando...")

    # Inicia thread
    t = threading.Thread(target=process_question, args=(question, placeholder_frame), daemon=True)
    t.start()

############################
# CONFIGURAÇÃO CustomTkinter
############################

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Marvel Chat")
app.geometry("600x700")

pdf_path = "data/herois_marvel.pdf"

# Carrega o logo
try:
    logo_image = Image.open("marvel_logo.png")
    desired_width = 200
    w_orig, h_orig = logo_image.size
    ratio = h_orig / w_orig
    new_height = int(desired_width * ratio)
    logo_image = logo_image.resize((desired_width, new_height), Image.Resampling.LANCZOS)
    logo_ctk = ctk.CTkImage(light_image=logo_image, dark_image=logo_image, size=(desired_width, new_height))
    logo_label = ctk.CTkLabel(app, text="", image=logo_ctk)
    logo_label.pack(pady=10)
except:
    pass

chat_frame = ctk.CTkScrollableFrame(app, width=500, height=400, corner_radius=10)
chat_frame.pack(padx=10, pady=(0,10), fill="both", expand=True)

entry_frame = ctk.CTkFrame(app)
entry_frame.pack(fill="x", padx=10, pady=5)

question_var = ctk.StringVar()
question_entry = ctk.CTkEntry(
    entry_frame,
    textvariable=question_var,
    width=400,
    placeholder_text="Digite sua pergunta..."
)
question_entry.pack(side="left", padx=(0,5), pady=5, fill="x", expand=True)
question_entry.bind("<Return>", send_question)

send_button = ctk.CTkButton(
    entry_frame,
    text="Enviar",
    command=send_question,
    fg_color="#ED1D24",
    hover_color="#C70000",
    corner_radius=8
)
send_button.pack(side="left", pady=5)

app.mainloop()
