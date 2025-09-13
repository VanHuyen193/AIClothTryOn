from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import io


from .inference import load_model_once, predict_image


app = FastAPI(title="SegFormer Semantic Segmentation")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/images", StaticFiles(directory="app/images"), name="images")
templates = Jinja2Templates(directory="app/templates")


@app.on_event("startup")
async def startup_event():
    # Load model once when the server starts
    load_model_once(model_dir="model")


@app.get("/segment", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("segment.html", {"request": request})

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read uploaded image bytes
    img_bytes = await file.read()
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")


    # Run prediction -> returns a PIL image of overlay or mask
    out_img = predict_image(pil_img)


    # Stream back as PNG
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")