# Hosting a Fine-Tuned Qwen Model on a VPS Using vLLM

## 1. Pick a VPS with a GPU

vLLM is built for GPU inference (it uses PagedAttention + continuous batching,

both of which assume CUDA). CPU serving is possible but too slow for a

production astrology-chat use case, so the VPS must have an NVIDIA GPU.

Rough VRAM guide for the fine-tuned model:

| Model | bf16 | AWQ/GPTQ (4-bit) |

|------------------|-----------|------------------|

| Qwen2.5-1.5B | ~4 GB | ~1.5 GB |

| Qwen2.5-7B | ~16 GB | ~5–6 GB |

| Qwen2.5-14B | ~30 GB | ~10 GB |

Providers that offer GPU VPS/instances: RunPod, Lambda Labs, Vultr Cloud GPU,

Hetzner (via partners), AWS EC2 (g5/g6 family), Paperspace, OVHcloud. For a

7B model, an RTX A5000/A6000, L4, or A10G class GPU (24 GB) is a comfortable

choice.

## 2. Base OS setup

sudo apt update \&\& sudo apt upgrade -y

sudo apt install -y build-essential git curl ufw
Enable a firewall early:

sudo ufw allow OpenSSH

sudo ufw allow 443/tcp     # HTTPS (via Nginx, later)

sudo ufw enable
## 3. Install the NVIDIA driver + CUDA

Most GPU VPS images (RunPod, Lambda) come with drivers preinstalled — verify

with nvidia-smi. On a bare Ubuntu box:

sudo apt install -y nvidia-driver-550

sudo reboot

nvidia-smi   # confirm the GPU + driver are visible after reboot
vLLM ships CUDA runtime dependencies via its pip wheel, so a separate CUDA

toolkit install is usually not required — the driver is enough.

## 4. Python environment

sudo apt install -y python3.11 python3.11-venv

python3.11 -m venv /opt/vedaz-venv

source /opt/vedaz-venv/bin/activate

pip install --upgrade pip

pip install vllm
Confirm the install detects the GPU:

python -c "import torch; print(torch.cuda.is\_available(), torch.cuda.get\_device\_name(0))"
## 5. Get the fine-tuned model onto the box

Copy the **merged** checkpoint (base weights + LoRA merged into one set of

weights — see the fine-tuning script) to the VPS, e.g.:

scp -r qwen-vedaz-astrologer-merged/ user@your-vps:/opt/models/
If VRAM is tight, quantize the merged model first (AWQ is the best supported

path for vLLM):

pip install autoawq

python -m awq.entry --model\_path /opt/models/qwen-vedaz-astrologer-merged \\

&#x20;   --quant\_path /opt/models/qwen-vedaz-astrologer-awq \\

&#x20;   --w\_bit 4 --q\_group\_size 128
## 6. Launch vLLM's OpenAI-compatible server

python -m vllm.entrypoints.openai.api\_server \\

&#x20;   --model /opt/models/qwen-vedaz-astrologer-merged \\

&#x20;   --served-model-name vedaz-astrologer \\

&#x20;   --host 0.0.0.0 --port 8000 \\

&#x20;   --max-model-len 4096 \\

&#x20;   --gpu-memory-utilization 0.90 \\

&#x20;   --api-key "$VLLM\_API\_KEY"
Key flags:

- --gpu-memory-utilization: how much of the GPU's memory vLLM is allowed to

pre-allocate for the KV cache (0.85–0.95 is typical).

- --max-model-len: caps context length; lower this if VRAM is limited.

- --quantization awq (or gptq) if serving a quantized checkpoint.

- --tensor-parallel-size N if the VPS has multiple GPUs and the model needs

to be sharded across them.

Test it:

curl http://localhost:8000/v1/chat/completions \\

&#x20; -H "Authorization: Bearer $VLLM\_API\_KEY" \\

&#x20; -H "Content-Type: application/json" \\

&#x20; -d '{

&#x20;   "model": "vedaz-astrologer",

&#x20;   "messages": \[{"role": "user", "content": "Mera career kaisa rahega is saal?"}]

&#x20; }'
## 7. Keep it running: systemd service

\# /etc/systemd/system/vllm-astrologer.service

\[Unit]

Description=vLLM Vedaz Astrologer server

After=network.target



\[Service]

User=vllm

WorkingDirectory=/opt/models

Environment="VLLM\_API\_KEY=your-secret-key"

ExecStart=/opt/vedaz-venv/bin/python -m vllm.entrypoints.openai.api\_server \\

&#x20;   --model /opt/models/qwen-vedaz-astrologer-merged \\

&#x20;   --served-model-name vedaz-astrologer \\

&#x20;   --host 0.0.0.0 --port 8000 \\

&#x20;   --max-model-len 4096 \\

&#x20;   --gpu-memory-utilization 0.90 \\

&#x20;   --api-key ${VLLM\_API\_KEY}

Restart=always

RestartSec=5



\[Install]

WantedBy=multi-user.target
sudo systemctl daemon-reload

sudo systemctl enable --now vllm-astrologer

sudo systemctl status vllm-astrologer
## 8. Put it behind Nginx + TLS

Don't expose vLLM's port directly to the internet — put Nginx in front for

TLS termination, rate limiting, and request logging.

\# /etc/nginx/sites-available/vedaz-api

server {

&#x20;   listen 443 ssl;

&#x20;   server\_name api.yourapp.com;



&#x20;   ssl\_certificate     /etc/letsencrypt/live/api.yourapp.com/fullchain.pem;

&#x20;   ssl\_certificate\_key /etc/letsencrypt/live/api.yourapp.com/privkey.pem;



&#x20;   location / {

&#x20;       proxy\_pass http://127.0.0.1:8000;

&#x20;       proxy\_set\_header Host $host;

&#x20;       proxy\_set\_header X-Real-IP $remote\_addr;

&#x20;       proxy\_read\_timeout 300s;   # LLM responses can be slow to stream

&#x20;   }

}
sudo ln -s /etc/nginx/sites-available/vedaz-api /etc/nginx/sites-enabled/

sudo certbot --nginx -d api.yourapp.com

sudo systemctl reload nginx
Close direct external access to port 8000 in ufw so all traffic must go

through Nginx.

## 9. Monitoring & operations

- nvidia-smi -l 5 — live GPU memory/utilization.

- journalctl -u vllm-astrologer -f — live server logs.

- vLLM exposes Prometheus metrics at /metrics — wire these into

Grafana/Prometheus for latency, throughput, and queue-depth dashboards.

- Set up log rotation and a simple health-check (/health endpoint or a

scripted curl + alert) so downtime is caught quickly.

## 10. Production hardening notes

- **Quantization** (AWQ/GPTQ/FP8) buys headroom on smaller VPS GPUs at a

small quality cost — worth benchmarking against the astrology use case

specifically (tone/empathy quality, not just perplexity).

- **Concurrency**: vLLM's continuous batching handles many simultaneous

chats efficiently on one GPU; still load-test with the expected concurrent

user count before launch.

- **Guardrails stay at the app layer**: even with a well fine-tuned model,

keep the crisis/self-harm redirect and "no guaranteed outcome" checks as a

thin rule-based layer in front of/behind the model, not solely relying on

learned behavior.

- **Versioning**: keep the merged checkpoint and the exact LoRA/base commit

hashes recorded so a bad fine-tune can be rolled back to the previous

--served-model-name quickly.
