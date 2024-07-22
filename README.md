# EasyRAG

## Objective
Develop a simple Retrieval-Augmented Generation (RAG) application that integrates a document retrieval system with a large language model (LLM) to generate responses based on retrieved documents.

## Requirements
- Python 3.11
- FastAPI
- Uvicorn
- OpenAI API key

## Setup Instructions

1. **Clone the repository:**

   ```sh
   git clone https://github.com/gorecki/easyrag.git
   cd easyrag
   ```

2. **Create and activate a virtual environment:**

   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages:**

   ```sh
   pip install -r requirements.txt
   ```

4. **Set up the OpenAI API key:**

   Ensure you have the OpenAI API key set in your environment variables:

   ```sh
   export OPENAI_API_KEY='your-api-key'  # On Windows use `set OPENAI_API_KEY=your-api-key`
   ```

5. **Run the application:**

   ```sh
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

6. **Test the API:**

   You can use the provided `test.http` file to test the API using an HTTP client.

## Usage Instructions

The API endpoint `/query` accepts a JSON payload with user input and returns a response generated based on relevant retrieved documents.

### Example request:

```json
POST /query
{
  "user_input": "Why can't we sell products in Italy and Spain?"
}
```

### Example response:

```json
{
  "response": "Your generated response here...",
  "retrieved_used": "Retrieved documents used in response...",
  "retrieved_not_used": "Retrieved documents not used in response..."
}
```


**Note:** The response processing splits the retrieved document 'no_sale_countries.md' into three distinct categories: retrieved and used, retrieved and not used, and not retrieved, directly contributing to the generated response. This allows the user to see the precision and relevancy of both the retrieval model and the LLM query. By providing the retrieved parts of the document used in the response, users can verify the accuracy and ensure the LLM does not hallucinate, thereby enhancing trust in the generated content.


## Files

- `main.py`: Main application code.
- `no_sale_countries.md`: Document containing no-sale countries information.
- `system.prompt`: System prompt for the language model.
- `test.http`: HTTP test file.

## Project Structure

```
easyrag/
├── .gitignore
├── main.py
├── no_sale_countries.md
├── README.md
├── requirements.txt
├── system.prompt
└── test.http
```

## Detailed Explanation

### main.py

The main application code. It integrates FastAPI for creating API endpoints, a document retrieval system using ChromaDB, and a language model from OpenAI to generate responses based on retrieved documents.

### no_sale_countries.md

This file contains the document with information about countries where sales are not conducted. It is used to retrieve relevant paragraphs based on the user's query.

### system.prompt

This file contains the system prompt used by the language model to generate responses. It ensures the model follows a specific structure while generating responses.

### test.http

A sample HTTP request file to test the API. It contains example requests to the `/query` endpoint.

## Installation of Specific Versions

Ensure that `chromadb` is installed with version 0.5.3 by specifying it in the `requirements.txt` file:

```
fastapi
uvicorn
openai
sentence-transformers
chromadb==0.5.3
```

## Limitations

1. **Embeddings Model Choice:**
   - The 'all-MiniLM-L6-v2' model has been used for generating embeddings due to its balance of performance and computational efficiency, making it suitable for many practical applications.

2. **LLM Selection:**
   - OpenAI's 'gpt-4o-mini' model was chosen for its favorable performance-to-cost ratio. Other models, such as those highlighted in the Chatbot Arena [leaderboard](https://chat.lmsys.org/?leaderboard), could also be considered based on specific requirements and budget constraints.

3. **Document Splitting:**
   - Splitting the retrieved document by '\n\n' is an ad hoc approach and could be refined further to ensure better segmentation and contextual relevance of the extracted paragraphs.

4. **Retrieval Results:**
   - The parameter `n_results=3` used during document retrieval is also an ad hoc choice. This setting could be optimized based on the application's specific needs and performance criteria.
