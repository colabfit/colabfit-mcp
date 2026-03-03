# colabfit-mcp

An MCP server for discovering [ColabFit](https://materials.colabfit.org) datasets and training MACE interatomic potentials.

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
mkdir -p ./colabfit_data/models ./colabfit_data/datasets ./colabfit_data/inference_output

# Or custom location (must match COLABFIT_DATA_ROOT in .env)
mkdir -p /your/custom/path/{models,datasets,inference_output}
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

## Tools

| Tool | Description |
|------|-------------|
| `search_datasets` | Search ColabFit database by text, elements, properties, software |
| `check_local_datasets` | Scan local data directory for downloaded datasets, filter by elements/properties |
| `download_dataset` | Download dataset as XYZ files with automatic analysis |
| `fine_tune_mace` | Fine-tune MACE-MP-0 foundation model on a dataset (recommended) |
| `train_mace` | Train a MACE model from scratch |
| `use_model` | Run energy/forces/stress/relax calculations with a trained model, or generate a Python snippet |
| `check_status` | Check GPU, packages, disk, existing models and datasets |

## Typical Workflow

1. `search_datasets` — find datasets with the elements/properties you need
2. `download_dataset` — download and auto-analyze for training suitability
3. `fine_tune_mace` — fine-tune the MACE-MP-0 foundation model on your data
4. `use_model` — run calculations or generate a reusable Python script

## Sample Prompts

The following prompts work directly in Claude Code or Claude Desktop once the MCP server is registered.

**Explore available data:**

> Search ColabFit for silicon datasets that include forces. Which ones look best for training an interatomic potential?

> What datasets do I have downloaded locally? Do any contain iron with stress data?

**End-to-end training:**

> Find a dataset for copper, download it, and fine-tune MACE-MP-0 on it. Use default settings.

> I need a potential for lithium phosphate. Search ColabFit for Li and P datasets, pick the most suitable one, and start fine-tuning.

**Run inference:**

> Use my model at colabfit_data/models/cu_mace/cu_mace_stagetwo.model to calculate the energy and forces on bulk copper in FCC structure.

> Relax an FCC aluminum structure with my trained model and report the final energy and cell parameters.

> Generate a Python snippet to run Langevin molecular dynamics on bulk silicon using my MACE model.

**Check status:**

> Check my GPU status and list all the models and datasets I have locally.

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

Training writes log files in your data directory under each model's subdirectory:

```bash
./colabfit_data/models/<model_name>/training.log
```

## GPU Support

`make start` automatically detects your GPU via `start.sh`:

- **NVIDIA GPU present**: starts with `compose.nvidia.yaml` overlay, enabling CUDA passthrough via nvidia-container-toolkit
- **No NVIDIA GPU**: starts without the overlay; the container selects the best available device (MPS or CPU) automatically at runtime

The pip-installed version handles GPU detection purely in Python via `detect_device()` — no shell wrapper needed, since PyTorch can see the host GPU directly.

## Local Installation (without Docker)

### Install

```bash
pip install colabfit-mcp                  # search, download, check_status only
pip install 'colabfit-mcp[train]'         # + MACE training (any CUDA version)
pip install 'colabfit-mcp[full]'          # + CUDA 12 optimized cuequivariance ops
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
├── mace-torch
├── MACE-MP-0 foundation (cached at build time)
└── Training via mace_run_train
```

Container managed by Docker Compose:
- **server** — MCP server + ML training

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `COLABFIT_DATA_ROOT` | `./colabfit_data` | Host-side bind-mount source directory. Inside the container the data root is always `/home/mcpuser/colabfit`. |
| `USER_ID` | `1000` | User ID for container (should match host user) |
| `GROUP_ID` | `1000` | Group ID for container (should match host user) |
| `FOUNDATION_MODEL` | `small` | MACE-MP-0 foundation size: `small`, `medium`, or `large` |
| `MACE_DTYPE` | `float32` | Training precision. Use `float64` only for geometry optimization. |
| `MACE_BATCH_SIZE` | `8` (fine-tune) / `16` (train) | Training batch size. Decrease if OOM. |
| `MACE_VALID_BATCH_SIZE` | `16` / `32` | Validation batch size (can be larger than training). |
| `MACE_NUM_WORKERS` | `4` | DataLoader worker processes for parallel data loading. |
| `COLABFIT_BASE_URL` | `https://materials.colabfit.org` | ColabFit API base URL |
| `COLABFIT_AUTH_USER` | `mcp-tool` | ColabFit API auth username |
| `COLABFIT_AUTH_PASS` | `mcp-secret` | ColabFit API auth password |

**Data Storage:**

By default, models and datasets are stored in `./colabfit_data/` (relative to the
project root), making data portable with the project. `COLABFIT_DATA_ROOT` controls
only the **host-side** bind-mount source — the container-internal data root is always
`/home/mcpuser/colabfit` regardless of this setting. To use a fixed host location that
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

## Manual Usage: Using a Trained Model with ASE

After training or fine-tuning, the model directory contains several `.model` files.
Use `<model_name>_stagetwo.model` for inference — it is the SWA-averaged final model
and generally has the best accuracy. This file is only produced when SWA is enabled
(the default). If SWA was disabled, use `<model_name>.model` instead. The `use_model`
tool's `model_path` parameter must point to the specific `.model` file.

### Attaching a MACE Calculator to ASE Atoms

```python
from mace.calculators import MACECalculator
from ase.build import bulk

calc = MACECalculator(
    model_paths="colabfit_data/models/<model_name>/<model_name>_stagetwo.model",
    device="cuda",       # or "cpu"
    default_dtype="float64",
)

atoms = bulk("Si", "diamond", a=5.43)
atoms.calc = calc
```

### Properties MACE Implements

| Method | Returns | Units | Notes |
|--------|---------|-------|-------|
| `atoms.get_potential_energy()` | scalar | eV | |
| `atoms.get_forces()` | `(N, 3)` array | eV/Å | |
| `atoms.get_stress()` | `(6,)` Voigt array | eV/Å³ | periodic structures only; see note below |
| `atoms.get_potential_energies()` | `(N,)` array | eV | per-atom energies |
| `atoms.calc.get_property("node_energy", atoms)` | `(N,)` array | eV | per-atom energies relative to atomic references; no `atoms.get_*` wrapper |
| `atoms.get_stresses()` | `(N, 6)` array | eV/Å³ | requires `compute_atomic_stresses=True` on calculator init |

Enable per-atom stresses at calculator creation:

```python
calc = MACECalculator(..., compute_atomic_stresses=True)
```

**Stress note:** Stress is always computed via autodiff (virial: ∂E/∂ε). When
`cauchy_stress` was present in the training data, the model was explicitly optimized
to reproduce those values and stress predictions will be accurate. Without stress
training data, the computation still runs but the values are less reliable.
Stress is not computed for non-periodic structures (no cell/PBC) and will raise
`PropertyNotImplementedError` in that case.

### Properties Computed by ASE Itself (no calculator needed)

| Method | Notes |
|--------|-------|
| `atoms.get_kinetic_energy()` | from `momenta` array; returns 0.0 if momenta not set |
| `atoms.get_temperature()` | derived from kinetic energy |
| `atoms.get_velocities()` | derived from momenta and masses |
| `atoms.get_total_energy()` | `get_potential_energy() + get_kinetic_energy()` |

### Unsupported Properties

Calling `atoms.get_charges()`, `atoms.get_magnetic_moments()`, or
`atoms.get_dipole_moment()` with a MACE calculator raises
`PropertyNotImplementedError` — MACE does not predict these.

### Common Workflows

**Geometry optimization:**

```python
from ase.optimize import BFGS

opt = BFGS(atoms, trajectory="relax.traj")
opt.run(fmax=0.01)  # converge forces below 0.01 eV/Å
```

**Molecular dynamics:**

```python
from ase.md.langevin import Langevin
from ase import units

dyn = Langevin(atoms, timestep=1.0 * units.fs, temperature_K=300, friction=0.01)
dyn.run(1000)
```

**Vibrational frequencies** (finite differences of forces; no stress training data needed):

```python
from ase.vibrations import Vibrations

vib = Vibrations(atoms)
vib.run()
vib.summary()
```

[mcp-name: io.github.colabfit/colabfit-mcp]: #
