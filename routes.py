from fastapi import APIRouter
from controller import router as ml_router

router  = APIRouter()

# Incluir el path de los routes
router.include_router(ml_router)