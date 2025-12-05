from pathlib import Path

# Carpeta raíz del proyecto (ajusta si hiciera falta)
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_RAW = BASE_DIR / "data" / "raw" 
DATA_PROCESSED = BASE_DIR / "data" / "processed"
INDEX_DIR = BASE_DIR / "index"

# Fichero final con todos los documentos chunked
DOCUMENTS_FILE = DATA_PROCESSED / "documentos.jsonl"

# Parámetros de chunking
CHUNK_SIZE = 800      # tamaño del trozo (caracteres aprox)
CHUNK_OVERLAP = 200   # solapamiento entre trozos
