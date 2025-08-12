from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pipeline_plus import run_pipeline  # 같은 폴더에 pipeline_plus.py가 있어야 함

class In(BaseModel):
    youtube_link: str

app = FastAPI(title="YouTube Summarizer API")

@app.post("/summarize")
def summarize(body: In):
    try:
        return run_pipeline(body.youtube_link)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

