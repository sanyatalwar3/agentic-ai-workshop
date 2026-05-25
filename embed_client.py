import os
import requests

from dotenv import load_dotenv

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

BASE = "https://integrate.api.nvidia.com/v1"

HEADERS = {
    "Authorization": f"Bearer {NVIDIA_API_KEY}",
    "Content-Type": "application/json",
}

MODEL = "nvidia/nv-embedqa-e5-v5"


def get_embeddings(texts, input_type="passage"):

    payload = {
        "model": MODEL,
        "input": texts,
        "input_type": input_type,
    }

    response = requests.post(
        f"{BASE}/embeddings",
        headers=HEADERS,
        json=payload,
    )

    data = response.json()

    return [item["embedding"] for item in data["data"]]