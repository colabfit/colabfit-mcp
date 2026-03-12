# colabfit-mcp

An MCP server for discovering [ColabFit](https://materials.colabfit.org) datasets and training MACE interatomic potentials using [KLIFF](https://kliff.readthedocs.io/) and [KLAY](https://github.com/openkim/klay).

## Setup

### Quick Start (Recommended)

```bash
git clone https://github.com/colabfit/colabfit-mcp.git
cd colabfit-mcp

# One-time setup: creates data directories and .env file
make setup

# Build Docker images with your user ID for proper permissions
make build

# Start with GPU detection (CUDA → CPU fallback)
make start
```

Run `make help` to see all available commands.

### Manual Setup

If you prefer not to use the Makefile:

#### 1. Configure environment

```bash
cp example.env .env
# Edit .env to customize data directory location if desired
```

#### 2. Create data directories

```bash
# Default location
mkdir -p ./colabfit_data/models ./colabfit_data/datasets ./colabfit_data/inference_output ./colabfit_data/test_driver_output

# Or custom location (must match COLABFIT_DATA_ROOT in .env)
mkdir -p /your/custom/path/{models,datasets,inference_output,test_driver_output}
```

#### 3. Build with user ID mapping

```bash
# This ensures the container user matches your host user for proper permissions
USER_ID=$(id -u) GROUP_ID=$(id -g) docker compose build
```

### 4. Register the MCP server

`start.sh` automatically detects NVIDIA GPU availability and enables GPU passthrough when present, falling back to CPU otherwise.

**Claude Code:**

```bash
claude mcp add colabfit-mcp -- /path/to/colabfit-mcp/start.sh
```

Replace `/path/to/colabfit-mcp` with the absolute path to this repository.
Then restart Claude Code for the new server to take effect.

**Claude Desktop:**

Add to your Claude Desktop config (`Settings > Developer > Edit Config`):

```json
{
  "mcpServers": {
    "colabfit-mcp": {
      "command": "/path/to/colabfit-mcp/start.sh",
      "args": ["run", "--rm", "-i", "server"]
    }
  }
}
```

### Generic MCP Client Setup

The server uses standard MCP `stdio` transport and works with any MCP-compatible client.

**Entry point** (after pip install or in the Docker container):

```bash
colabfit-mcp          # registered console script
# or
python -m colabfit_mcp
```

**Testing with mcp-cli:**

```bash
pip install mcp-cli
mcp-cli run colabfit-mcp -- colabfit-mcp
```

**Any stdio MCP client** can register the server using the same `command` / `args` pattern as Claude Desktop above. All tools use standard MCP `stdio` transport — no HTTP server or open port is required.

> Note: Docker is required for training and inference (heavy dependencies). The `search_datasets`, `check_local_datasets`, `download_dataset`, and `check_status` tools work without Docker via a plain pip install.

## Tools

| Tool | Description |
|------|-------------|
| `search_datasets` | Search ColabFit database by text, elements, properties, software |
| `check_local_datasets` | Scan local data directory for downloaded datasets, filter by elements/properties |
| `download_dataset` | Download a dataset from HuggingFace via KLIFF |
| `train_mace` | Train a MACE-style KLAY model from scratch using KLIFF |
| `use_model` | Run energy/forces/relax calculations with a trained KLAY model, or generate a Python snippet |
| `check_status` | Check GPU, packages, disk, existing models and datasets |
| `list_test_drivers` | List available kimvv test drivers, optionally filtered by property keyword |
| `run_test_driver` | Run a kimvv test driver against a trained KLAY model; saves `structures.extxyz` + `results.json` in a timestamped subdirectory; supports multiple structures per call with optional `repeat` for supercell sizing and `async_mode` for slow drivers |
| `check_test_driver_result` | Check status of an async test driver job and return inline results when complete |

### Available Test Drivers (kimvv)

| Test Driver | Description | Properties |
|---|---|---|
| `EquilibriumCrystalStructure` | Equilibrium lattice parameters and cohesive energy | lattice-constant, cohesive-energy |
| `ElasticConstantsCrystal` | Full elastic constants tensor at zero temperature | elastic-constants |
| `CrystalStructureAndEnergyVsPressure` | Crystal structure and energy as a function of pressure | energy-vs-pressure |
| `GroundStateCrystalStructure` | Lowest energy crystal structure among candidates | ground-state-structure |
| `VacancyFormationEnergyRelaxationVolumeCrystal` | Vacancy formation energy and relaxation volume | vacancy-formation-energy, relaxation-volume |
| `ClusterEnergyAndForces` | BFGS relaxation of an atomic cluster in a non-periodic box. Use for molecular/non-periodic models. | energy, atomic-forces, relaxed-positions |

## Typical Workflow

1. `search_datasets` — find datasets with the elements/properties you need
2. `download_dataset` — download from HuggingFace (cached locally for reuse)
3. `train_mace` — train a MACE-style KLAY model on the downloaded data
4. `use_model` — run energy/forces/relax calculations or generate a Python snippet
5. `run_test_driver` — validate the model against OpenKIM-style property tests

## Sample Prompts

The following prompts work directly in Claude Code or Claude Desktop once the MCP server is registered.

**Explore available data:**

> Search ColabFit for silicon datasets that include forces. Which ones look best for training an interatomic potential?

> What datasets do I have downloaded locally? Do any contain iron with stress data?

**End-to-end training:**

> Find a dataset for copper, download it, and train a MACE model on it. Use default settings.

> I need a potential for lithium phosphate. Search ColabFit for Li and P datasets, pick the most suitable one, and start training.

**Run inference:**

> Use my model at /home/mcpuser/colabfit/models/cu_mace/cu_mace__MO_000000000000_000 to calculate the energy and forces on bulk copper in FCC structure.

> Relax an FCC aluminum structure with my trained model and report the final energy and cell parameters.

> Generate a Python snippet to run the energy calculation on bulk silicon using my KLAY model.

**Validate with test drivers:**

> What test drivers are available for validating my model?

> Run the ElasticConstantsCrystal test driver on my silicon model at /home/mcpuser/colabfit/models/si_mace/si_mace__MO_000000000000_000.

> Run the EquilibriumCrystalStructure and VacancyFormationEnergyRelaxationVolumeCrystal tests on my copper FCC model.

**Check status:**

> Check my GPU status and list all the models and datasets I have locally.

**End-to-end workflow:**

> Search ColabFit for silicon datasets with forces, download the best one, train a MACE model, calculate energy and forces on bulk diamond-cubic silicon, then run the ElasticConstantsCrystal and EquilibriumCrystalStructure test drivers to validate the model. Report the elastic constants and equilibrium lattice parameter when done.

## Stopping / Canceling Training

The MCP server runs via `docker compose run` (not `docker compose up`), so
`docker compose down` alone will **not** stop an active training container.
Use the methods below to stop the server including any in-progress training job.

### Using Makefile

```bash
make stop
```

### Without Makefile

```bash
# Stop all containers belonging to this project (catches both 'up' and 'run' containers)
docker ps -q --filter "label=com.docker.compose.project=colabfit-mcp" | xargs -r docker stop
docker compose down
```

If the project directory is not named `colabfit-mcp`, replace the filter value with your
directory name (lowercased). You can check the label on a running container with:

```bash
docker inspect <container-id> --format '{{ index .Config.Labels "com.docker.compose.project" }}'
```

> Training progress is saved as `training.log` inside the model's KIM subdirectory
> (`<model_name>__MO_000000000000_000/training.log`). Stopping mid-training discards any
> in-progress epoch; completed epochs and their checkpoints are preserved on disk.

## Monitoring Training Progress

View training output in the following ways:

### 1. Real-time Container Logs (Recommended)

View live training output as it happens:

```bash
# Using Makefile
make logs

# Or directly with docker compose
docker compose logs -f server
```

Press `Ctrl+C` to exit (training continues in background).

### 2. Persistent Log Files

Training writes log files inside the model's KIM subdirectory:

```bash
./colabfit_data/models/<model_name>/<model_name>__MO_000000000000_000/training.log
```

## GPU Support

`make start` automatically detects your GPU via `start.sh`:

- **NVIDIA GPU present**: starts with `compose.nvidia.yaml` overlay, enabling CUDA passthrough via nvidia-container-toolkit
- **No NVIDIA GPU**: starts without the overlay; the container selects the best available device (MPS or CPU) automatically at runtime

The pip-installed version handles GPU detection purely in Python via `detect_device()` — no shell wrapper needed, since PyTorch can see the host GPU directly.

## Local Installation (without Docker)

### Install

```bash
pip install colabfit-mcp                  # search, check_status only
pip install 'colabfit-mcp[full]'          # + KLIFF/KLAY training, HuggingFace download
```

### Register with Claude Code

```bash
claude mcp add colabfit-mcp -- colabfit-mcp
```

### Register with Claude Desktop

Add to your Claude Desktop config (`Settings > Developer > Edit Config`):

```json
{
  "mcpServers": {
    "colabfit-mcp": {
      "command": "colabfit-mcp"
    }
  }
}
```

### Data directory

By default, datasets and models are stored under `~/colabfit/`. Override with:

```bash
export COLABFIT_DATA_ROOT=/your/preferred/path
```

Subdirectories are created automatically the first time each tool writes data.

### Requirements

- Python 3.10+
- CUDA 12.x + nvidia drivers (for GPU training; CPU fallback works without CUDA)

## Architecture

```
server container
├── MCP server (FastMCP, stdio)
├── KLIFF (dataset loading, training orchestration)
├── KLAY (MACE-style model construction)
└── Training via KLIFF GNNLightningTrainer
```

Datasets are downloaded from HuggingFace (`colabfit/` org) as parquet/arrow files via KLIFF's
`Dataset.from_huggingface` and cached locally. Models are MACE-style graphs
built with KLAY and trained with KLIFF's Lightning trainer.

Container managed by Docker Compose:
- **server** — MCP server + ML training

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `COLABFIT_DATA_ROOT` | `./colabfit_data` | **Host directory** for datasets and models (bind mount) |
| `USER_ID` | `1000` | User ID for container (should match host user) |
| `GROUP_ID` | `1000` | Group ID for container (should match host user) |
| `KLIFF_BATCH_SIZE` | `4` | Training batch size. Decrease if OOM. |
| `KLIFF_NUM_WORKERS` | `0` | DataLoader worker processes. Keep at 0 to avoid CUDA fork deadlocks. |
| `TRAIN_SIZE` | `0` | Number of training configs (0 = auto 90% split) |
| `VAL_SIZE` | `0` | Number of validation configs (0 = auto 10% split) |
| `KLIFF_DTYPE` | `float32` | Training precision (`float32` default; use `float64` for higher accuracy) |
| `COLABFIT_BASE_URL` | `https://materials.colabfit.org` | ColabFit API base URL (used by search) |
| `COLABFIT_AUTH_USER` | `mcp-tool` | ColabFit API auth username (used by search) |
| `COLABFIT_AUTH_PASS` | `mcp-secret` | ColabFit API auth password (used by search) |

**Data Storage:**

By default, models and datasets are stored in `./colabfit_data/` (relative to the
project root), making data portable with the project. To use a fixed location that
persists across project clones, set `COLABFIT_DATA_ROOT` in `.env`:

```bash
cp example.env .env
# Edit .env and set: COLABFIT_DATA_ROOT=/home/yourusername/ml_data
```

**User ID Mapping:**

The `USER_ID` and `GROUP_ID` variables ensure the container user matches your host
user, preventing permission issues with bind-mounted directories. The Makefile
automatically detects your IDs, but you can override them in `.env` if needed.

## Requirements

- Docker with Compose v2
- NVIDIA GPU + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (optional; CPU fallback used automatically if absent)
- Or: Python 3.10+ for local installation

## Troubleshooting

**GPU not detected in container**: Ensure `nvidia-container-toolkit` is
installed and the Docker daemon has been restarted. Verify with
`docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi`.
If no NVIDIA GPU is present, use `./start.sh` which falls back to CPU automatically.

**MCP server not responding**: The server uses stdio transport, not HTTP. It
must be launched via `docker compose run --rm -i server`, not accessed
over a network port.

---

## Manual Usage: Running Inference with a Trained KLAY Model

After training, the model directory (`model_path` returned by `train_mace`) contains
`model.pt` and `kliff_graph.param`. Use these directly with PyTorch and KLIFF.

### Loading and Running the Model

```python
import numpy as np
import torch
from torch_scatter import scatter_add
from kliff.dataset import Configuration
from kliff.transforms.configuration_transforms.graphs.generate_graph import RadialGraph
from ase.build import bulk

atoms = bulk("Si", "diamond", a=5.43)

model_dir = "/home/mcpuser/colabfit/models/colabfit_mace/colabfit_mace__MO_000000000000_000"

# Load model (tries TorchScript first, falls back to torch.load)
try:
    model = torch.jit.load(f"{model_dir}/model.pt")
except Exception:
    model = torch.load(f"{model_dir}/model.pt", weights_only=False)
model.eval()

# Build graph — read species/cutoff from kliff_graph.param
transform = RadialGraph(species=["Si"], cutoff=5.0, n_layers=1)
config = Configuration(
    cell=atoms.cell.array,
    species=list(atoms.get_chemical_symbols()),
    coords=atoms.get_positions(),
    PBC=list(atoms.get_pbc()),
    energy=0.0,
    forces=np.zeros((len(atoms), 3)),
)
graph = transform(config)

device = "cuda" if torch.cuda.is_available() else "cpu"
coords = graph.coords.clone().detach().to(torch.float32).to(device).requires_grad_(True)
energy = model(
    species=graph.species.to(device),
    coords=coords,
    edge_index0=graph.edge_index0.to(device),
    contributions=graph.contributions.to(device),
)
print(f"Energy: {energy.sum().item():.4f} eV")

# Forces via autograd
(grad,) = torch.autograd.grad(energy.sum(), coords)
forces = -scatter_add(grad, graph.images.to(device), dim=0)[:len(atoms)]
print(f"Forces (eV/Å):\n{forces.detach().cpu().numpy()}")
```

### Geometry Optimization with ASE

The `use_model` tool's `_KliffInlineCalculator` wraps the KLAY model as an ASE
calculator. For custom scripts, replicate the same pattern:

```python
from ase.optimize import BFGS

# (attach _KliffInlineCalculator from use_model module, or replicate the pattern)
opt = BFGS(atoms, trajectory="relax.traj")
opt.run(fmax=0.01)  # converge forces below 0.01 eV/Å
```

[mcp-name: io.github.colabfit/colabfit-mcp]: #
