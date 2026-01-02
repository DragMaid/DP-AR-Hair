from fastapi import FastAPI
from routers import (
    admin,
    assignment,
    worker,
    task
)

app = FastAPI()


app.include_router(admin.router)
app.include_router(worker.router)
app.include_router(task.router)
app.include_router(assignment.router)
