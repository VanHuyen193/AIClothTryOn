LOCAL RUN
python -m venv .venv
. .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

Local Run with Docker
docker build -t segformer-web:latest .
docker run --rm -p 8000:80 segformer-web:latest