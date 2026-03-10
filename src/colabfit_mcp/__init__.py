import logging

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
    from colabfit_mcp.tools.train import train_mace
    from colabfit_mcp.tools.use_model import use_model
    from colabfit_mcp.tools.create_structure import create_structure
    mcp.tool()(train_mace)
    mcp.tool()(use_model)
    mcp.tool()(create_structure)
except ImportError as e:
    logger.warning(f"Training/inference tools disabled: {e}. Install with [full] extras.")

try:
    from colabfit_mcp.tools.test_driver import list_test_drivers, run_test_driver
    mcp.tool()(list_test_drivers)
    mcp.tool()(run_test_driver)
except ImportError as e:
    logger.warning(f"Test driver tools disabled: {e}. Install with [full] extras.")


def main():
    mcp.run(transport="stdio")
