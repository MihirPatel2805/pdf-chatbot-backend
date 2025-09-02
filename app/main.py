from fastapi import FastAPI

app = FastAPI(title="PDF Chatbot Backend")


@app.get("/")
def read_root():
    return {"message": "API is running"}


