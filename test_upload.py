from fastapi import FastAPI, File, UploadFile
from typing import List
import shutil
import tempfile
import os

app = FastAPI()

# Create a temporary directory
temp_dir = tempfile.TemporaryDirectory()

@app.post("/uploadfiles/")
async def upload_files(files: List[UploadFile] = File(...)):
    upload_dir = temp_dir.name

    for file in files:
        file_location = os.path.join(upload_dir, file.filename)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    return {"filenames": [file.filename for file in files], "directory": upload_dir}
