# colabfit-mcp

An MCP server for discovering [ColabFit](https://materials.colabfit.org) datasets, training MACE interatomic potentials, and deploying models via OpenKIM/KIM-API.

## Setup

### Quick Start (Recommended)

```bash
git clone https://github.com/colabfit/colabfit-mcp.git
cd colabfit-mcp

# One-time setup: creates data directories and .env file
make setup

# Build Docker images with your user ID (avoids permission issues)
make build

# Start all services
make start

# View logs (optional)
make logs
```

Run `make help` to see all available commands.

### Manual Setup

If you prefer not to use the Makefile:

#### 1. Configure environment

```bash
cp .env.example .env
# Edit .env to customize data directory location if desired
```

#### 2. Create data directories

```bash
# Default location
mkdir -p ./colabfit_data/models ./colabfit_data/datasets

# Or custom location (must match COLABFIT_DATA_ROOT in .env)
mkdir -p /your/custom/path/{models,datasets}
```

#### 3. Build with user ID mapping

```bash
# This ensures the container user matches your host user (avoids permission issues)
USER_ID=$(id -u) GROUP_ID=$(id -g) docker compose build
```

#### 4. Start MongoDB

```bash
docker compose up -d mongodb
```

Wait a few seconds, then confirm it's healthy:

```bash
docker compose ps
```

You should see `mongodb` with status `Up ... (healthy)`.

### 5. Register the MCP server

**Claude Code:**

```bash
claude mcp add colabfit-mcp -- docker compose -f /path/to/colabfit-mcp/compose.yaml run --rm -i colabfit-mcp
```

Replace `/path/to/colabfit-mcp` with the absolute path to this repository.
Then restart Claude Code for the new server to take effect.

**Claude Desktop:**

Add to your Claude Desktop config (`Settings > Developer > Edit Config`):

```json
{
  "mcpServers": {
    "colabfit-mcp": {
      "command": "docker",
      "args": [
        "compose", "-f", "/path/to/colabfit-mcp/compose.yaml",
        "run", "--rm", "-i", "colabfit-mcp"
      ]
    }
  }
}
```

### After a reboot or `docker compose down`

The MCP server registration persists in your Claude config, but the MongoDB
container does not auto-start. After a reboot, you only need to restart MongoDB:

```bash
cd /path/to/colabfit-mcp
docker compose up -d mongodb
```

Then start (or restart) Claude Code or Claude Desktop. You do **not** need to
re-register the MCP server.

If you ran `docker compose down`, the same applies -- just `docker compose up -d mongodb`
and restart your client. All trained models and downloaded datasets are stored in
bind-mounted directories on the host filesystem (default: `./colabfit_data/`) and
survive both reboots and `docker compose down`.

## Tools

| Tool | Description |
|------|-------------|
| `search_datasets` | Search ColabFit database by text, elements, properties, software |
| `download_dataset` | Download dataset as XYZ files with automatic analysis |
| `fine_tune_mace` | Fine-tune MACE-MP-0 foundation model on a dataset (recommended) |
| `train_mace` | Train a MACE model from scratch |
| `deploy_model` | Export to TorchScript and install as KIM Portable Model |
| `check_status` | Check GPU, packages, MongoDB, disk, existing models |

## Typical Workflow

1. `search_datasets` -- find datasets with the elements/properties you need
2. `download_dataset` -- download and auto-analyze for training suitability
3. `fine_tune_mace` -- fine-tune the MACE-MP-0-a medium foundation model
4. `deploy_model` -- export and install as a KIM Portable Model

## Monitoring Training Progress

Training output is available in **two ways** for maximum visibility:

### 1. Real-time Container Logs (Recommended)

View live training output as it happens:

```bash
# Using Makefile
make logs

# Or directly with docker compose
docker compose logs -f colabfit-mcp
```

Press `Ctrl+C` to exit (training continues in background).

### 2. Persistent Log Files

Training also writes to log files in your data directory for historical review:

```bash
# Default data directory
tail -f ./colabfit_data/models/training.log

# Custom COLABFIT_DATA_ROOT location (if set in .env)
tail -f $COLABFIT_DATA_ROOT/models/training.log

# Model-specific logs (when available)
ls -la ./colabfit_data/models/*/logs/
```

**For Claude Code Users:**

Claude can troubleshoot training issues by:
- Reading log files directly from the host filesystem
- Viewing container logs via `docker compose logs` commands
- Monitoring training metrics and loss curves

**Benefits:**
- ✅ Real-time visibility into training progress
- ✅ Persistent logs survive container restarts
- ✅ Easy to share logs for debugging
- ✅ Claude can directly analyze training issues

## Local Installation (without Docker)

```bash
pip install colabfit-mcp            # Base: search + download only
pip install 'colabfit-mcp[train]'   # + MACE training
pip install 'colabfit-mcp[full]'    # + kimpy, kliff, KIM deployment
```

## Architecture

```
colabfit-mcp container (GPU)     mongodb container
├── MCP server (FastMCP, stdio)  └── KIMKit model store
├── mace-torch, kimpy, kliff
├── kim-api
├── MACE-MP-0-a medium (cached)
└── Training via mace_run_train
```

Two containers managed by Docker Compose:
- **colabfit-mcp** -- MCP server + ML training + KIM packages (GPU-enabled)
- **mongodb** -- MongoDB 8.0 for KIMKit model management

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `COLABFIT_DATA_ROOT` | `./colabfit_data` | **Host directory** for datasets and models (bind mount) |
| `USER_ID` | `1000` | User ID for container (should match host user) |
| `GROUP_ID` | `1000` | Group ID for container (should match host user) |
| `COLABFIT_BASE_URL` | `https://materials.colabfit.org` | ColabFit API base URL |
| `COLABFIT_AUTH_USER` | `mcp-tool` | ColabFit API auth username |
| `COLABFIT_AUTH_PASS` | `mcp-secret` | ColabFit API auth password |
| `MONGODB_HOST` | `mongodb` | MongoDB hostname (internal) |
| `MONGODB_PORT` | `27017` | MongoDB port |

**Data Storage:**

By default, models and datasets are stored in `./colabfit_data/` (relative to the
project root), making data portable with the project. To use a fixed location that
persists across project clones, set `COLABFIT_DATA_ROOT` in `.env`:

```bash
cp .env.example .env
# Edit .env and set: COLABFIT_DATA_ROOT=/home/yourusername/ml_data
```

**User ID Mapping:**

The `USER_ID` and `GROUP_ID` variables ensure the container user matches your host
user, preventing permission issues with bind-mounted directories. The Makefile
automatically detects your IDs, but you can override them in `.env` if needed.

## Requirements

- Docker with Compose v2
- NVIDIA GPU + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (for training)
- Or: Python 3.10+ for local installation

## Troubleshooting

**MCP server fails to connect / times out**: The most common cause is MongoDB
not running when the MCP server starts. The server can auto-start MongoDB, but
the health check adds ~12 seconds of startup time, which may exceed your
client's connection timeout. Fix: run `docker compose up -d mongodb` and wait
for it to be healthy before starting your client. With MongoDB already running,
the server starts in ~2 seconds.

**GPU not detected in container**: Ensure `nvidia-container-toolkit` is
installed and the Docker daemon has been restarted. Verify with
`docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi`.

**MongoDB connection failures**: The entrypoint waits up to 30 seconds for
MongoDB. If it doesn't connect, check that the `mongodb` service is healthy
with `docker compose ps`. The MCP server will still start but KIMKit features
will be unavailable.

**MCP server not responding**: The server uses stdio transport, not HTTP. It
must be launched via `docker compose run --rm -i colabfit-mcp`, not accessed
over a network port.

## License

The ColabFit Tools package is copyrighted by the Regents of the University of
Minnesota. It can be freely used for educational and research purposes by
non-profit institutions and US government agencies only. Other organizations are
allowed to use the ColabFit Tools package only for evaluation purposes, and any
further uses will require prior approval. The software may not be sold or
redistributed without prior approval. One may make copies of the software for
their use provided that the copies, are not sold or distributed, are used under
the same terms and conditions. As unestablished research software, this code is
provided on an "as is" basis without warranty of any kind, either expressed or
implied. The downloading, or executing any part of this software constitutes an
implicit agreement to these terms. These terms and conditions are subject to
change at any time without prior notice.

[mcp-name: io.github.colabfit/colabfit-mcp]: #
