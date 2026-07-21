FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app app
COPY requirements-admin.txt ./
RUN pip install --no-cache-dir -r requirements-admin.txt
COPY --chown=app:app . .

USER app
EXPOSE 8001
CMD ["uvicorn", "admin_app:app", "--host", "0.0.0.0", "--port", "8001", "--proxy-headers"]
