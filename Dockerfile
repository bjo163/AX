FROM python:3.9

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . /app
RUN rm -rf /app/ui

EXPOSE 8001/tcp

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]