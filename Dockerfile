FROM python:3.9

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt


COPY ./main.py /app/


CMD ["fastapi", "run", "main.py", "--port", "80"]