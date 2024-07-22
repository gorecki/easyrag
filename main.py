"""
A simple Retrieval-Augmented Generation (RAG) application that integrates a document retrieval system with a large language model (LLM) to generate responses based on retrieved documents.
"""

import os
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb

# Check if the OPENAI_API_KEY environment variable is set
def check_openai_api_key():
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise EnvironmentError("The OPENAI_API_KEY environment variable is not set.")

# Call the checker function at the start and initialize OpenAI API client
check_openai_api_key()
client = OpenAI() 

def load_text_file(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as file:
        return file.read()

class Embeddings:
    ''' 
    Prepares embeddings for the given document using a predefined model and saves it to a predefined 
    file if the file does not exist yet. 
    Otherwise, embeddings are just loaded from the file previously created file. 
    '''
    def __init__(self, doc_file_name):
        self.emb_model_name = 'all-MiniLM-L6-v2'
        self.model = SentenceTransformer(self.emb_model_name)
        self.doc_file_name = doc_file_name
        self.client = chromadb.EphemeralClient() 
        self.collection_name = "embeddings"
        self.collection = self.client.get_or_create_collection(name=self.collection_name) # https://github.com/langchain-ai/langchain/issues/24163

    def prepare(self):
        self.document = load_text_file(self.doc_file_name)
        self.paragraphs = self.document.split("\n\n")

        # Split the document
        embeddings = self.model.encode(self.paragraphs)

        # Add embeddings to Chroma DB
        for i, emb in enumerate(embeddings):
            self.collection.add(ids=[str(i)], embeddings=[emb.tolist()], documents=[self.paragraphs[i]])

        print('Embeddings have been added to Chroma DB.')



class Query(BaseModel):
    user_input: str


class RAG:
    def __init__(self, emb):
        self.emb = emb
        self.system_prompt = load_text_file('system.prompt')

    def retrieve_documents(self, query: str):
        query_embedding = self.emb.model.encode([query])[0]
        results = self.emb.collection.query(query_embeddings=[query_embedding.tolist()], n_results=3)
        return results

    def generate_response(self, ret_documents: dict, query: str):
        joined_docs = ''
        for i, doc in enumerate(ret_documents['documents'][0]):
            joined_docs += f'<p id="{ret_documents["ids"][0][i]}">{doc}</p>\n\n'
        prompt = f"User query: {query}\n\nRelevant documents:\n{joined_docs}\n\nResponse:"
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
        )        
        return response.choices[0].message.content
    
    def parse_p_used(self, p_used_string):
        # Regular expression to extract the list of ids
        pattern = r'<p-used>\[(.*?)\]</p-used>'
        match = re.search(pattern, p_used_string)
        
        if match:
            # Extract the matched group and split it into a list of ids
            ids_string = match.group(1)
            ids_list = ids_string.replace('\'', '').split(',')
            # Strip any extra whitespace from the ids
            ids_list = [id.strip() for id in ids_list]
            return ids_list
        else:
            return []
        
    def format_response(self, ret_documents, response):
        # Get the retrieved paragraphs that are indeed used in the response
        paragraphs_used = rag.parse_p_used(response)
        
        # Cut off the <p-used> tag
        fresponse = response.split('<p-used>')[0]
        
        # Classify retrieved paragraphs according to its usage in the response
        retrieved_used, retrieved_not_used = '', ''
        for i, rd in enumerate(ret_documents['documents'][0]):
            if ret_documents['ids'][0][i] in paragraphs_used:
                retrieved_used += rd + '\n\n'
            else:
                retrieved_not_used += rd + '\n\n'

        return {"response": fresponse,
                "retrieved_used": retrieved_used,
                "retrieved_not_used": retrieved_not_used
                }        


# Initialize FastAPI app
app = FastAPI()

# Prepare embeddings for a given text file
emb = Embeddings('no_sale_countries.md')
emb.prepare()

# Initialize RAG
rag = RAG(emb)


@app.post("/query")
async def get_response(query: Query):
    ret_documents = rag.retrieve_documents(query.user_input)
    if not ret_documents:
        raise HTTPException(status_code=404, detail="No relevant documents found")
    response = rag.generate_response(ret_documents, query.user_input)
    return rag.format_response(ret_documents, response)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
