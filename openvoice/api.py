import os
import sys
import json
import uvicorn
import traceback
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from starlette.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from inference.infer_tool_v2 import Svc  # Asegúrate de que infer_tool_v2.py existe y tiene la clase Svc
from inference import infer_tool_v2  # Asegúrate de que infer_tool_v2.py existe

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/voices")
async def get_available_voices():
    voice_list = os.listdir(Svc.model_path)
    return JSONResponse(content={"voices": voice_list})


@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    try:
        file_location = f"tmp/{file.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())
        return {"info": "file uploaded successfully", "filename": file.filename}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": "File upload failed"})


@app.post("/clone_voice")
async def clone_voice(file: UploadFile = File(...), voice_name: str = "default"):
    try:
        # Guardar archivo de audio temporalmente
        file_location = f"tmp/{file.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())

        # Llamar al modelo para clonar la voz
        svc = Svc()
        clone_result, embeddings, mel_spectrogram = svc.clone_voice(file_location, voice_name)

        # Guardar embeddings y espectrogramas
        np.save(f"embeddings/{voice_name}_embeddings.npy", embeddings)
        np.save(f"spectrograms/{voice_name}_mel_spectrogram.npy", mel_spectrogram)

        # Visualizar y guardar los espectrogramas
        plt.figure(figsize=(10, 4))
        plt.imshow(mel_spectrogram, aspect='auto', origin='lower')
        plt.colorbar()
        plt.title(f'Mel Spectrogram - {voice_name}')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.savefig(f'spectrograms/{voice_name}_mel_spectrogram.png')
        plt.close()

        # Eliminar el archivo temporal
        os.remove(file_location)

        return JSONResponse(content={"clone_result": clone_result})
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": "Voice cloning failed"})


@app.post("/generate_speech")
async def generate_speech(text: str, voice_name: str = "default"):
    try:
        svc = Svc()
        speech_result = svc.generate_speech(text, voice_name)
        return JSONResponse(content={"speech_result": speech_result})
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": "Speech generation failed"})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
