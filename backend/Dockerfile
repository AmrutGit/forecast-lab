FROM python:3.11-slim

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ml ./ml
COPY backend ./backend

RUN chmod +x backend/entrypoint.sh

ENV PYTHONPATH=/srv

EXPOSE 8000

CMD ["backend/entrypoint.sh"]
