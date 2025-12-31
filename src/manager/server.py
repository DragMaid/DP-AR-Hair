from fastapi import FastAPI
from manager.database import list_tasks
from typing import Optional

app = FastAPI()


@app.get("/tasks/")
def get_tasks(limit: Optional[int] = 100):
    return {"tasks": list_tasks(limit)}
