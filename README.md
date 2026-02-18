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

# Start all services
make start

# View logs
make logs
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
mkdir -p ./colabfit_data/models ./colabfit_data/datasets

# Or custom location (must match COLABFIT_DATA_ROOT in .env)
mkdir -p /your/custom/path/{models,datasets}
```

#### 3. Build with user ID mapping

```bash
# This ensures the container user matches your host user for proper permissions
USER_ID=$(id -u) GROUP_ID=$(id -g) docker compose build
```

### 4. Register the MCP server

**Claude Code:**

```bash
claude mcp add colabfit-mcp -- docker compose -f /path/to/colabfit-mcp/compose.yaml run --rm -i server
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
        "run", "--rm", "-i", "server"
      ]
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
| `deploy_model` | Export to TorchScript and install as KIM Portable Model |
| `check_status` | Check GPU, packages, disk, existing models |

## Typical Workflow

1. `search_datasets` -- find datasets with the elements/properties you need
2. `download_dataset` -- download and auto-analyze for training suitability
3. `fine_tune_mace` -- fine-tune the MACE-MP-0 foundation model on your data
4. `deploy_model` -- export and install as a KIM Portable Model

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

Training writes log files in your data directory under each model's subdirectory, i.e.:

```bash
./colabfit_data/models/<model_name>/training.log

```

## Using a Trained Model with ASE

After training or fine-tuning, the model directory contains several `.model` files.
Use `<model_name>_stagetwo.model` for inference — it is the SWA-averaged final model
and generally has the best accuracy.

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

## Local Installation (without Docker)

```bash
pip install colabfit-mcp            # Base: search + download only
pip install 'colabfit-mcp[full]'    # MACE Training
```

## Architecture

```
server container (GPU)             
├── MCP server (FastMCP, stdio)    
├── mace-torch
├── kim-api (Not yet implemented)
├── MACE-MP-0 foundation (cached)
└── Training via mace_run_train
```

Container managed by Docker Compose:
- **server** -- MCP server + ML training + KIM packages (GPU-enabled)

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `COLABFIT_DATA_ROOT` | `./colabfit_data` | **Host directory** for datasets and models (bind mount) |
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
- NVIDIA GPU + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (for training)
- Or: Python 3.10+ for local installation

## Troubleshooting

**GPU not detected in container**: Ensure `nvidia-container-toolkit` is
installed and the Docker daemon has been restarted. Verify with
`docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi`.

**MCP server not responding**: The server uses stdio transport, not HTTP. It
must be launched via `docker compose run --rm -i server`, not accessed
over a network port.

[mcp-name: io.github.colabfit/colabfit-mcp]: #
