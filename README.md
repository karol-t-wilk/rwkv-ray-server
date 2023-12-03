# RWKV Ray Server

## Dependencies

```bash
pip install "ray[serve]" rwkv torch fastapi pydantic
```

## Run

```bash
serve run config.yml
```

This will start the ray deployment locally at `localhost:8000` by default. Depending on the size of the model
you downloaded, it can take quite a while to start the server.

## Usage

The server at this point only accepts one endpoint, `POST /generate`. Example:

```bash
curl -X POST -H "Content-type: application/json" -N -d '{"prompt": "Question: What is a good prompt for the RWKV 4 World model?\nAnswer:", "max_tokens": 256}' http://localhost:8000/generate
```
