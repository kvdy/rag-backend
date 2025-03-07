Setup Instructions

Prerequisites
Python 3.8+: Ensure you have Python installed. You can download it from python.org.
PostgreSQL: Ensure you have PostgreSQL installed and running. You can download it from postgresql.org.
Ensure pgvector extension is installed in your PostgreSQL database.

Installation
Clone the repository:



git clone https://github.com/kvdy/rag-backend
cd rag-backend

Create a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

Install dependencies:



Set up environment variables:

Create a .env file in the root directory of the project.

Add the following environment variables to the .env file:
OPENAI_API_KEY="your-openai-api-key"
DB_NAME="your-database-name"
DB_USER="your-database-user"
DB_PASSWORD="your-database-password"
DB_HOST="your-database-host"
DB_PORT=your-database-port

Database Setup
Create the database:

psql -U postgres
CREATE DATABASE your_db_name;

Create the necessary tables:

Running the Application
Start the FastAPI server:

uvicorn main:app --reload

Access the API documentation:

Open your browser and navigate to http://127.0.0.1:8000/docs to view the interactive API documentation.
Usage
Upload a file:

Use the /upload/ endpoint to upload files.
Ingest documents:

Use the /ingest/ endpoint to ingest documents from the specified directory.
Query the ingested documents:

Use the /query/ endpoint to query the ingested documents.

Testing
Run the provided test cases:
Use the commands in the tests.txt file to test the various endpoints.

Additional Information
The application uses FastAPI for the backend.
The documents are processed and stored in a PostgreSQL database.
The application uses OpenAI for embeddings and language model responses.