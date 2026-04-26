from __future__ import annotations

import logging
from typing import Annotated, Any, Dict, Optional, Sequence

from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from app.agents.prompts import system_prompt
from app.agents.tools import restaurant_tools

logger = logging.getLogger("edu_agent")


class ChatResponse(BaseModel):
    """사용자에게 최종 응답을 포맷팅합니다."""

    message_id: str = Field(description="UUID 형식의 고유 메시지 ID")
    content: str = Field(description="사용자 질문에 대한 답변")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="추가 메타데이터"
    )


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def get_restaurant_agent(model, checkpointer):
    schema_tools = restaurant_tools + [ChatResponse]
    executable_tools = restaurant_tools

    model_with_tools = model.bind_tools(schema_tools)

    async def call_model(state: AgentState) -> dict:
        messages = list(state["messages"])
        response = await model_with_tools.ainvoke(
            [SystemMessage(content=system_prompt)] + messages
        )
        logger.debug("call_model response: %s", type(response).__name__)
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        tool_calls: list = getattr(state["messages"][-1], "tool_calls", [])
        if not tool_calls:
            return END
        first_name: str = tool_calls[0].get("name", "")
        logger.debug("should_continue: tool_call=%s", first_name)
        return END if first_name == "ChatResponse" else "tools"

    tool_node = ToolNode(executable_tools)

    graph = StateGraph(AgentState)
    graph.add_node("model", call_model)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("model")
    graph.add_conditional_edges(
        "model",
        should_continue,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "model")

    logger.info("Restaurant agent compiled successfully")
    return graph.compile(checkpointer=checkpointer)
