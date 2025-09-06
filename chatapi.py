import os
from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import base64

load_dotenv()

my_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=my_api_key)

class ChatRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    response: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins; restrict in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/")
def ai_prompt(request: ChatRequest):
    response = client.responses.create(
    model="gpt-5-mini",
    input=request.prompt
    )

    gpt_response = response.output_text
    return ChatResponse(response=gpt_response)


@app.post("/uploadfile/")
async def create_upload_file(prompt: str = Form(...), file: UploadFile = File(None)):
    base64_image = None
    response = None
    if file:
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode("utf-8")

        response = client.responses.create(
            model="gpt-4.1",
            input=[
                {
                    "role": "user",
                    "content": [
                        { "type": "input_text", "text": prompt },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    ],
                }
            ],
        )
    else:
        response = client.responses.create(
            model="gpt-5-mini",
            input=prompt
        )
    if response:  
        gpt_response = response.output_text
        print("GPT RESPONSE", gpt_response)
        return ChatResponse(response=gpt_response)
    return {"message": "No file uploaded"}