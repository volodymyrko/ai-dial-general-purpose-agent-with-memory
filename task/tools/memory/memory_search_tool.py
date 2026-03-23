import json
from typing import Any

from task.tools.base import BaseTool
# from task.tools.memory._models import MemoryData
from task.tools.memory.memory_store import LongTermMemoryStore
from task.tools.models import ToolCallParams


class SearchMemoryTool(BaseTool):
    """
    Tool for searching long-term memories about the user.

    Performs semantic search over stored memories to find relevant information.
    """

    def __init__(self, memory_store: LongTermMemoryStore):
        self.memory_store = memory_store


    @property
    def name(self) -> str:
        # TODO: provide self-descriptive name
        # raise NotImplementedError()
        return 'search_long_term_memory'

    @property
    def description(self) -> str:
        # TODO: provide tool description that will help LLM to understand when to use this tools and cover 'tricky'
        #  moments (not more 1024 chars)
        # raise NotImplementedError()
        return (
            "Search important information about the user in long-term memory. "
            "Use this to search user preferences, personal details, goals, and context "
        )

    @property
    def parameters(self) -> dict[str, Any]:
        # TODO: provide tool parameters JSON Schema:
        #  - query is string, description: "The search query. Can be a question or keywords to find relevant memories", required
        #  - top_k is integer, description: "Number of most relevant memories to return.", minimum is 1, maximum is 20, default is 5
        # raise NotImplementedError()
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Can be a question or keywords to find relevant memories."
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of most relevant memories to return",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 20
                }
            },
            "required": ["query"]
        }


    async def _execute(self, tool_call_params: ToolCallParams) -> str:
        #TODO:
        # 1. Load arguments with `json`
        # 2. Get `query` from arguments
        # 3. Get `top_k` from arguments, default is 5
        # 4. Call `memory_store` `search_memories` (we will implement logic in `memory_store` later)
        # 5. If results are empty then set `final_result` as "No memories found.",
        #    otherwise iterate through results and collect content, category and topics (if preset) in markdown format
        # 6. Add result to stage as markdown text
        # 7. Return result
        # raise NotImplementedError()
        args = json.loads(tool_call_params.tool_call.function.arguments)
        query = args['query']
        top_k = args.get('top_k', 5)

        result = await self.memory_store.search_memories(
            api_key=tool_call_params.api_key,
            query=query,
            top_k=top_k
        )

        if not result:
            final_result = 'No memories found.'
        else:
            final_result = f"Found {len(result)} relevant memories:\n"
            for memory in result:
                final_result += f"**Category:**{memory.category},\n **Importance:**{memory.importance},\n"
                if memory.topics:
                    final_result += f"**Topics:** {', '.join(memory.topics)}, \n"
                final_result += f"**Content:**{memory.content};\n\n"
        
        tool_call_params.stage.append_content(final_result)

        return final_result
