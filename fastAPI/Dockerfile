FROM python:3.11

WORKDIR /app
COPY . /app
WORKDIR /app/app
RUN pip install pandas
RUN pip install -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port=8080"]
