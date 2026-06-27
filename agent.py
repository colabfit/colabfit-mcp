import asyncio
from agents import Agent, Runner, SQLiteSession
from agents.mcp import MCPServerStreamableHttp

from dotenv import load_dotenv
import os
load_dotenv()
from pathlib import Path
DATA_ROOT = Path(os.environ.get("COLABFIT_DATA_ROOT", str(Path.home() / "colabfit")))

SESSION_DB = DATA_ROOT / "agent_sessions.sqlite"

MCP_URL = os.environ.get("COLABFIT_MCP_URL", "http://localhost:8000/mcp")


AGENT_INSTRUCTIONS = """
You are a ColabFit MCP task-execution agent.

Your job is to complete the user's requested materials-science pipeline, not merely discuss it.

Use the ColabFit MCP tools for dataset queries, dataset download, file inspection,
model training, simulation execution, and pipeline actions.

Tool order constraints:
- The valid training workflow is:
  1. search_datasets or check_local_datasets
  2. download_dataset if needed
  3. train_mace
  4. run_test_driver only after train_mace succeeds
- Never call run_test_driver with a dataset directory.
- Never call run_test_driver unless a previous train_mace call returned a model path.
- If check_local_datasets returns "Use train_mace", the next tool call must be train_mace.
- The phrase "I will now train" is forbidden. Either call train_mace in the same run or report that train_mace is unavailable.

Core rules:
1. Do not invent tool results. If a tool is needed, call the appropriate MCP tool.
2. Do not ask the user to repeat information already present in the conversation.
3. Treat short confirmations like "yes", "ok", "proceed", "please do", or "do it"
   as permission to continue the immediately previous task.
4. If the user request contains multiple pipeline steps, execute all steps in order
   without asking for confirmation between steps unless the action is destructive.
5. If the user names a local dataset and asks to train a model, call the training tool.
6. For training from an existing dataset, use the structures already in the dataset.
   Do not ask for FCC, diamond, lattice constant, or structure unless the specific
   training tool explicitly requires it and the information cannot be inferred.
7. For a silicon dataset, infer element = Si.
8. Keep track of the active task state:
   - selected dataset
   - selected dataset path
   - selected element/system
   - output directory
   - current pipeline stage
   - trained model path, if available
9. If a tool fails, report the exact failure and the next concrete recovery step.
10. Your final answer for a pipeline action must include:
   - dataset used
   - tool/action performed
   - output path or model path, if available
   - success/failure status

Avoid educational detours unless the user explicitly asks for explanation.
"""


async def main():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    session = SQLiteSession(
        "colabfit_chat_001",
        db_path=str(SESSION_DB),
    )

    async with MCPServerStreamableHttp(
        params={
            "url": MCP_URL,
            "timeout": 30,
            "sse_read_timeout": 6000,
        },
        client_session_timeout_seconds=1800,
        cache_tools_list=True,
        name="colabfit-mcp",
    ) as server:

        agent = Agent(
            name="ColabFitAgent",
            model=os.environ.get("OPENAI_AGENT_MODEL", "gpt-4.1-mini"),
            instructions=AGENT_INSTRUCTIONS,
            mcp_servers=[server],
        )

        print(f"\nMCP Agent connected to: {MCP_URL}")
        print(f"Session DB: {SESSION_DB}")
        print("Type 'exit' to stop")

        while True:
            user_input = await asyncio.to_thread(input, "You: ")

            if user_input.lower().strip() in {"exit", "quit"}:
                break

            try:
                result = await Runner.run(
                    agent,
                    user_input,
                    session=session,
                    max_turns=20,
                )

            except Exception as e:
                print(f"\nError: {type(e).__name__}: {e}\n")
                continue

            print("\n--- Debug: new run items ---")
            for item in result.new_items:
                print(type(item).__name__)
                print(item)
                print()
            print("\nAgent:", result.final_output, "\n")


if __name__ == "__main__":
    asyncio.run(main())
