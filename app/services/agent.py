from pydantic_ai import Agent
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

agent = Agent(
    'google-gla:gemini-1.5-flash',
    # tools=[duckduckgo_search_tool()],
    system_prompt='Be concise, reply with one sentence.',
)
