from fastapi import FastAPI
from app.api.import_routes import router as import_router
from app.core.logging import logger

app = FastAPI(title='Dataset Import API', version='0.1.0')
app.include_router(import_router)

@app.get('/health', tags=['health'])
async def health():
    return {'status': 'ok'}
