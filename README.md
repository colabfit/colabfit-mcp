# colabfit-mcp

An MCP server for interacting with the [ColabFit](https://materials.colabfit.org) database.

## Installation
Download source code:
```
git clone https://github.com/colabfit/colabfit-mcp.git
```
Make sure the mcp package is installed.
```
pip install mcp[cli]
```
Modify your Claude configuration file.
Navigate to Settings &rarr; Developer &rarr; Edit Config
Your config file should look something like:
```
    {
    "mcpServers": {
        "query-mcp": {
        "command": "<path to environment's mcp command>",
        "args": [
            "run",
            "<download-location>/colabfit-mcp/colabfit_mcp.py>"
        ]
        }
    }
        }
```
## License
The ColabFit Tools package is copyrighted by the Regents of the University of
Minnesota. It can be freely used for educational and research purposes by
non-profit institutions and US government agencies only. Other organizations are
allowed to use the ColabFit Tools package only for evaluation purposes, and any
further uses will require prior approval. The software may not be sold or
redistributed without prior approval. One may make copies of the software for
their use provided that the copies, are not sold or distributed, are used under
the same terms and conditions. As unestablished research software, this code is
provided on an "as is'' basis without warranty of any kind, either expressed or
implied. The downloading, or executing any part of this software constitutes an
implicit agreement to these terms. These terms and conditions are subject to
change at any time without prior notice.
