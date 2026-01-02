from fastapi import APIRouter

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[],
)


@router.get("/login")
def authenticate_admin(username: str, password: str):
    pass


@router.get("/reset")
def reset_admin_account():
    pass
