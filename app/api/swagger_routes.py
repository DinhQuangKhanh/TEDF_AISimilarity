# Optional Swagger shortcut route

from fastapi import APIRouter
from fastapi.responses import RedirectResponse

router = APIRouter(tags=['swagger'])

@router.get('/swagger', include_in_schema=False)
def swagger_redirect():
    return RedirectResponse(url='/docs')
