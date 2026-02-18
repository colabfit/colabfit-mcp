from mcp.server.fastmcp import FastMCP

from colabfit_mcp.tools.search import search_datasets
from colabfit_mcp.tools.local_datasets import check_local_datasets
from colabfit_mcp.tools.download import download_dataset
from colabfit_mcp.tools.fine_tune import fine_tune_mace
from colabfit_mcp.tools.train import train_mace
from colabfit_mcp.tools.deploy import deploy_model
from colabfit_mcp.tools.status import check_status

mcp = FastMCP("colabfit-mcp")

mcp.tool()(search_datasets)
mcp.tool()(check_local_datasets)
mcp.tool()(download_dataset)
mcp.tool()(fine_tune_mace)
mcp.tool()(train_mace)
mcp.tool()(deploy_model)
mcp.tool()(check_status)


def main():
    mcp.run(transport="stdio")
