from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from app.api.analyze_routes import router as analyze_router
from app.api.import_routes import router as import_router
from app.api.thesis_routes import router as thesis_router
from app.api.similarity_routes import router as similarity_router
from app.core.logging import logger

app = FastAPI(title='Dataset Import API', version='0.1.0')

# Dev-friendly CORS: lets the demo page (opened via /demo, a Live Server, or file://) call the API
# even when it is not exactly same-origin. No credentials are used, so '*' is safe here.
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(import_router)
app.include_router(thesis_router)
app.include_router(similarity_router)
app.include_router(analyze_router)

@app.get('/', include_in_schema=False)
async def root():
    return RedirectResponse(url='/docs')

@app.get('/health', tags=['health'])
async def health():
    return {'status': 'ok'}
