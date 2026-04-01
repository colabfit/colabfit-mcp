import logging
import warnings


def _configure_logging():
    logging.getLogger("httpx").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", message=".*unauthenticated.*HF Hub.*")
    try:
        from loguru import logger as _loguru_logger
        _loguru_logger.disable("kliff.transforms.configuration_transforms.descriptors")
    except ImportError:
        pass


_configure_logging()

from mcp.server.fastmcp import FastMCP

from colabfit_mcp.tools.search import search_datasets
from colabfit_mcp.tools.local_datasets import check_local_datasets
from colabfit_mcp.tools.download import download_dataset
from colabfit_mcp.tools.status import check_status

logger = logging.getLogger(__name__)

mcp = FastMCP("colabfit-mcp")

mcp.tool()(search_datasets)
mcp.tool()(check_local_datasets)
mcp.tool()(download_dataset)
mcp.tool()(check_status)

try:
    from colabfit_mcp.tools.build_dataset import build_dataset
    mcp.tool()(build_dataset)
except ImportError as e:
    logger.warning(f"build_dataset tool disabled: {e}. Install numpy, pyarrow, and ase.")

try:
    from colabfit_mcp.tools.train import train_mace
    from colabfit_mcp.tools.use_model import use_model
    from colabfit_mcp.tools.create_structure import create_structure
    mcp.tool()(train_mace)
    mcp.tool()(use_model)
    mcp.tool()(create_structure)
except ImportError as e:
    logger.warning(f"Training/inference tools disabled: {e}. Install with [full] extras.")

try:
    from colabfit_mcp.tools.test_driver import check_test_driver_result, list_test_drivers, run_test_driver
    mcp.tool()(list_test_drivers)
    mcp.tool()(run_test_driver)
    mcp.tool()(check_test_driver_result)
except ImportError as e:
    logger.warning(f"Test driver tools disabled: {e}. Install with [full] extras.")


def main():
    mcp.run(transport="stdio")
