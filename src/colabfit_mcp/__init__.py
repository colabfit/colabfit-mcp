from mcp.server.fastmcp import FastMCP

from colabfit_mcp.tools.search import search_datasets
from colabfit_mcp.tools.local_datasets import check_local_datasets
from colabfit_mcp.tools.download import download_dataset
from colabfit_mcp.tools.train import train_mace
from colabfit_mcp.tools.use_model import use_model
from colabfit_mcp.tools.status import check_status

mcp = FastMCP("colabfit-mcp")

mcp.tool()(search_datasets)
mcp.tool()(check_local_datasets)
mcp.tool()(download_dataset)
mcp.tool()(train_mace)
mcp.tool()(use_model)
mcp.tool()(check_status)


def main():
    mcp.run(transport="stdio")
