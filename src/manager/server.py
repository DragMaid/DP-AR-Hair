from fastapi import FastAPI
from routers import (
    assignment,
    worker,
    task,
    auth
)
from core.exceptions import register_app_error_handler

app = FastAPI()


app.include_router(worker.router)
app.include_router(task.router)
app.include_router(assignment.router)
app.include_router(auth.router)

register_app_error_handler(app)
