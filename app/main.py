import os
from fastapi import FastAPI, File, Form, UploadFile
from dotenv import load_dotenv
from .vectorDB import convert_pdf_to_vector_db

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="PDF Chatbot Backend")


@app.get("/")
def read_root():
    return {"message": "API is running"}

@app.post('/upload-pdf')
async def upload_pdf(name: str = Form(...),
    description: str = Form(...),
    files: list[UploadFile] = File(...)):
    # Create upload directory
    upload_dir = f"./uploads/{name}"
    os.makedirs(upload_dir, exist_ok=True)  
    
    file_paths = []
    for file in files:
        file_location = os.path.join(upload_dir, file.filename)
        with open(file_location, "wb") as f:
            f.write(await file.read())
        file_paths.append(file_location)
    index_name = 'bot-01'           
    
    convert_pdf_to_vector_db(file_paths, index_name)
    print(f"PDF uploaded successfully: {file_paths}")
    return {"message": "PDF uploaded successfully"}
