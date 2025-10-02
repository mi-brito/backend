# Usar una imagen de Python
FROM python:3.9-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar el archivo de dependencias e instalarlas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar TODOS tus archivos de código (.py) al contenedor
COPY . .

# Exponer el puerto que usará el servidor web
EXPOSE 8000

# El comando para arrancar tu aplicación usando el "conector" main.py
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]