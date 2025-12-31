from fastapi import FastAPI
from manager.database import list_tasks, list_workers, list_assignments
from typing import Optional

app = FastAPI()


@app.get("/tasks/")
def get_tasks(limit: Optional[int] = 100):
    return {"tasks": list_tasks(limit)}


@app.get("/workers/")
def get_workers(limit: Optional[int] = 100):
    return {"workers": list_workers(limit)}


@app.get("/assignments/")
def get_assignments(limit: Optional[int] = 100):
    return {"assignments": list_assignments(limit)}
