# 首先，用户提交要处理的文档，该文档可以是PDF或图像格式。
# 第二个模块用于检测文件的格式，以便应用相关内容提取功能。
# 然后使用该模块将文档的内容分成多个块Data Splitter。
# Chunk Transformer这些块最终在存储到向量存储中之前使用 转换为嵌入。
# 在该过程结束时，用户的查询用于查找包含该查询答案的相关块，并将结果作为 JSON 返回给用户

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from param_config import  ParseConfig,ArgumentParser
from filetype import guess
from langchain.document_loaders.image import UnstructuredImageLoader
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter

from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from doc_analyze import DocAnalyze
from langchain.chains.question_answering import load_qa_chain
from translation_chain import TranslationChain,PDFTranslator
import tiktoken
import time
import gradio as gr


os.environ["TESSDATA_PREFIX"] = "C:\Program Files\Tesseract-OCR"

def detect_document_type(document_path):

    guess_file = guess(document_path)
    file_type = ""
    image_types = ['jpg', 'jpeg', 'png', 'gif']

    if(guess_file.extension.lower() == "pdf"):
        file_type = "pdf"

    elif(guess_file.extension.lower() in image_types):
        file_type = "image"

    else:
        file_type = "unkown"

    return file_type

def get_doc_search(text_splitter):

    enbeddings = OpenAIEmbeddings()
    print('text_splitter: %s',text_splitter)
    #return FAISS.from_texts(text_splitter, enbeddings)
    return FAISS.from_documents(text_splitter, enbeddings)

def extract_file_content(file_path):
    file_type = detect_document_type(file_path)

    if file_type == 'pdf':
        loader = UnstructuredFileLoader(file_path)
    elif file_type == "image":
        loader = UnstructuredImageLoader(file_path)
    
    documents_content = loader.load()
    #documents_content = '\n'.join(doc.page_content for doc in documents)
    # print(type(documents_content))
    return documents_content


def save_data_fasis(file_inputs,doc_class):

    print('file_path %s',file_inputs.name)
    print('doc class %s',doc_class)

    file_content = extract_file_content(file_inputs.name)
    print('file content %s',file_content)
    text_splitter =  CharacterTextSplitter (        
        separator = "\n\n",
        chunk_size =  150,
        chunk_overlap   =  0, 
    )
    text_splitter = text_splitter.split_documents(file_content)
    docana = DocAnalyze("text-davinci-003")
    print(docana.run(text_splitter))

    # documents_translate = ''

    # 翻译成英文
    # for sub_text in text_splitter:
    #     #encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    #     time.sleep(20)
    #     sub_text = ''.join(sub_text).replace('\n',' ')
    #     translator = PDFTranslator(config.model_name)
    #     documents_translate += translator.translate_pdf(sub_text)

    #print(text_splitter)

    document_search = get_doc_search(text_splitter)
    if doc_class == '老人与海':
        document_search.save_local("oldman_sea_index")
    elif doc_class == 'SQL 优化':
        document_search.save_local("sql_optimizer_index")
    elif doc_class == 'SQL Plan':
        document_search.save_local("sql_plan_index")
        

def chat_with_file(query,docclass):

    enbeddings = OpenAIEmbeddings()

    if docclass == '老人与海':
        document_search = FAISS.load_local("oldman_sea_index", enbeddings)
    elif docclass == 'SQL 优化':
        document_search = FAISS.load_local("sql_optimizer_index", enbeddings)
    elif docclass == 'SQL Plan':
        document_search = FAISS.load_local("sql_plan_index", enbeddings)

    print('docclass %s',docclass)

    documents = document_search.similarity_search(query)

    #answers = documents
    #print(documents[0].page_content)sss
    answers = ''
    for idx,doc in enumerate(documents):
        answers += documents[idx].page_content
    

    #调用 load_qa_chain

    # chain = load_qa_chain(OpenAI(temperature=0), chain_type = "map_rerank",  return_intermediate_steps=True)

    # #chain = load_qa_chain(ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0), chain_type = "map_rerank",  return_intermediate_steps=True)

    # results = chain({
    #                     "input_documents":documents, 
    #                     "question": query
    #                 }, 
    #                 return_only_outputs=True)
    # answers = results['intermediate_steps'][0]

    return answers

def launch_gradio():

    # 老版本的界面
    # iface = gr.Interface(
    #     fn=chat_with_pdf,
    #     title="OpenAI PDF对话工具",
    #     inputs=[
    #         # gr.File(label="上传PDF文件"),
    #         # gr.Textbox(label="源语言（默认：英文）", placeholder="English", value="English"),
    #         # gr.Textbox(label="目标语言（默认：中文）", placeholder="Chinese", value="Chinese")
    #         gr.Textbox(label="问题", value=" ")
    #     ],
    #     outputs="text",
    #     allow_flagging="never"
    # )
    with gr.Blocks() as iface:
        gr.Markdown("OpenAI PDF对话工具")
        
        with gr.Tab("向量转换"):
            with gr.Column():
                file_inputs=[
                gr.File(label="上传PDF文件"),
                gr.Dropdown(
                        ["SQL 优化", "老人与海","SQL Plan"],info="文档类别"
                            ),
                ]
                # allow_flagging="never"
                file_button=gr.Button("向量转换及存储")

        with gr.Tab("问题搜索"):
            with gr.Column():
                text_input=gr.Textbox(label="问题", value=" ")
                text_output=gr.Textbox(label="答案", value=" ")
                doc_class=gr.Dropdown(["SQL 优化", "老人与海","SQL Plan"],info="文档类别")
                ask_button=gr.Button("搜索...")
        
         
        ask_button.click(chat_with_pdf,[text_input,doc_class],outputs=text_output)     
        file_button.click(save_data_fasis,inputs=file_inputs)

    iface.launch(share=True, server_name="0.0.0.0")

def initialize_search_config():
    # 解析命令行
    argument_parser = ArgumentParser()
    args = argument_parser.parse_arguments()

    global config 
    config = ParseConfig()
    config.initialize(args)

def chat_with_pdf(query,docclass):
    # research_paper_path = config.input_file
    # print(research_paper_path)
    #save_data_fasis(research_paper_path)    
    #print(research_paper_path)

    results = chat_with_file(query,docclass)
    return results

if __name__ == "__main__":
   # 初始化 translator
    #initialize_search_config()
    # 启动 Gradio 服务
    launch_gradio()