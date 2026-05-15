import asyncio
from agents import Agent, Runner
from agents.mcp import MCPServerStreamableHttp

from dotenv import load_dotenv
import os
load_dotenv()

from pathlib import Path
DATA_ROOT = Path(os.environ.get("COLABFIT_DATA_ROOT", str(Path.home() / "colabfit")))
print(dataroot)
print(DATA_ROOT)

async def main():
    async with MCPServerStreamableHttp(
        params={
            "url": "http://127.0.0.1:8000/mcp",
            # "env":  mcp_env
        },
        client_session_timeout_seconds=6000,
        name="colabfit-mcp",
    ) as server:

        agent = Agent(
            name="ColabFitAgent",
            model="gpt-4.1-mini",
            instructions="""
            Use the tools to answer questions related to running a pipeline of materials-science simulations
            Do not answer the prompts directly.
            Always use the appropriate tool that matches the request.
            """,
            mcp_servers=[server],
        )

        print("\n MCP Agent")
        print("Type 'exit' to stop")
        while True:
            user_input = input("You:")

            if user_input.lower() == "exit": break

            result = await Runner.run(agent,user_input)

            print("\nAgent: ", result.final_output, "\n")

asyncio.run(main())
