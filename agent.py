import asyncio
from agents import Agent, Runner
from agents.mcp import MCPServerStdio


from dotenv import load_dotenv
import os
load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")
# print(api_key)

from pathlib import Path
dataroot = os.getenv("COLABFIT_DATA_ROOT")
DATA_ROOT = Path(os.environ.get("COLABFIT_DATA_ROOT", str(Path.home() / "colabfit")))
print(dataroot)
print(DATA_ROOT)

async def main():

    async with MCPServerStdio(

	# this is wrong. dont use the package like a script.
#         params={
#             "command": "python",
#             "args": ["-m", "colabfit_mcp"]
#        },

	# use start.sh as your entrypoint so as to startup the mcp-server
	params = {
		"command" : "/home/nmohan/UMN/colabfit-mcp/start.sh",
		"args": ["run", "--rm", "-i", "server"]
#		"command":"/home/nmohan/miniconda3/envs/openai-mcp/bin/colabfit-mcp"
#		"command":"/home/nmohan/.local/bin/colabfit-mcp"
	},
        client_session_timeout_seconds=6000
    ) as server:

        tools = await server.list_tools()
        # print("AVAILABLE TOOLS:", tools)

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

        result = await Runner.run(
            agent,
            "Search the ColabFit database and list all available models for Silicon."
        )

        print(result.final_output)

asyncio.run(main())
