import asyncio
import contextlib
from datetime import datetime
import json
from typing import Any, Optional
import uuid

from app.core.config import settings
from app.utils.logger import log_execution, custom_logger

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from pydantic import SecretStr

from app.agents.restaurant_agent import get_restaurant_agent


class AgentService:
    def __init__(self):
        self.model = ChatOpenAI(model=settings.OPENAI_MODEL, api_key=SecretStr(settings.OPENAI_API_KEY))
        self.checkpointer = MemorySaver()
        self.agent: Any = None
        self.progress_queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()

    def _create_agent(self, thread_id: uuid.UUID | None = None):
        """Restaurant LangGraph 에이전트 생성"""
        # Restaurant agent를 사용해 LangGraph 기반 에이전트를 초기화합니다.
        # model과 checkpointer를 전달하여 스트리밍 및 상태 저장을 지원합니다.
        self.agent = get_restaurant_agent(self.model, self.checkpointer)

    # 실제 대화 로직
    @log_execution
    async def process_query(self, user_messages: str, thread_id: uuid.UUID):
        """LangChain Messages 형식의 쿼리를 처리하고 AIMessage 형식으로 반환합니다."""
        try:
            # 에이전트 초기화 (한 번만)
            self._create_agent(thread_id=thread_id)

            custom_logger.info(f"사용자 메시지: {user_messages}")

            # IMP: LangGraph 에이전트에 사용자의 메시지를 HumanMessage 형태로 전달하고, 
            # thread_id를 통해 대화 문맥(Context)을 유지하며 비동기 스트리밍(astream)으로 실행하는 구현.
            agent_stream = self.agent.astream(
                {"messages": [HumanMessage(content=user_messages)]},
                config={"configurable": {"thread_id": str(thread_id)}},
                stream_mode="updates",
            )

            agent_iterator = agent_stream.__aiter__()
            agent_task = asyncio.ensure_future(agent_iterator.__anext__())
            progress_task = asyncio.ensure_future(self.progress_queue.get())

            while True:
                pending: set[asyncio.Future[Any]] = set()
                if agent_task is not None:
                    pending.add(agent_task)
                if progress_task is not None:
                    pending.add(progress_task)
                if not pending:
                    break

                done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

                if progress_task is not None and progress_task in done:
                    try:
                        progress_event = progress_task.result()
                        yield json.dumps(progress_event, ensure_ascii=False)
                        progress_task = asyncio.ensure_future(self.progress_queue.get())
                    except asyncio.CancelledError:
                        progress_task = None
                    except Exception as e:
                        # progress_task에서 예외 발생 시 로그만 남기고 계속 진행
                        custom_logger.error(f"Error in progress_task: {e}")
                        progress_task = None

                if agent_task is not None and agent_task in done:
                    try:
                        chunk = agent_task.result()
                    except StopAsyncIteration:
                        agent_task = None
                        break
                    except Exception as e:
                        # Task에서 발생한 예외 처리
                        custom_logger.error(f"Error in agent_task: {e}")
                        import traceback
                        custom_logger.error(traceback.format_exc())
                        agent_task = None
                        # 에러를 스트리밍으로 전송
                        error_response = {
                            "step": "done",
                            "message_id": str(uuid.uuid4()),
                            "role": "assistant",
                            "content": "처리 중 오류가 발생했습니다. 다시 시도해주세요.",
                            "metadata": {},
                            "created_at": datetime.utcnow().isoformat(),
                            "error": str(e)
                        }
                        yield json.dumps(error_response, ensure_ascii=False)
                        break

                    custom_logger.info(f"에이전트 청크: {chunk}")
                    try:
                        for step, event in chunk.items():
                            if not event or not (step in ["model", "tools"]):
                                continue
                            messages = event.get("messages", [])
                            if len(messages) == 0:
                                continue
                            message = messages[0]
                            if step == "model":
                                tool_calls = message.tool_calls
                                if not tool_calls:
                                    raw_content = getattr(message, "content", "") or ""
                                    if raw_content:
                                        done_event = self._parse_plain_text_response(raw_content)
                                        yield json.dumps(done_event, ensure_ascii=False)
                                    continue
                                tool = tool_calls[0]
                                if tool.get("name") == "ChatResponse":
                                    args = tool.get("args", {})
                                    metadata = args.get("metadata")
                                    custom_logger.info("========================================")
                                    custom_logger.info(args)
                                    yield f'{{"step": "done", "message_id": {json.dumps(args.get("message_id"))}, "role": "assistant", "content": {json.dumps(args.get("content"), ensure_ascii=False)}, "metadata": {json.dumps(self._handle_metadata(metadata), ensure_ascii=False)}, "created_at": "{datetime.utcnow().isoformat()}"}}'
                                else:
                                    yield f'{{"step": "model", "tool_calls": {json.dumps([tool["name"] for tool in tool_calls])}}}'
                            if step == "tools":
                                yield f'{{"step": "tools", "name": {json.dumps(message.name)}, "content": {message.content}}}'
                    except Exception as e:
                        # 청크 처리 중 예외 발생
                        custom_logger.error(f"Error processing chunk: {e}")
                        import traceback
                        custom_logger.error(traceback.format_exc())
                        error_response = {
                            "step": "done",
                            "message_id": str(uuid.uuid4()),
                            "role": "assistant",
                            "content": "데이터 처리 중 오류가 발생했습니다.",
                            "metadata": {},
                            "created_at": datetime.utcnow().isoformat(),
                            "error": str(e)
                        }
                        yield json.dumps(error_response, ensure_ascii=False)
                        break

                    agent_task = asyncio.ensure_future(agent_iterator.__anext__())

            if progress_task is not None:
                progress_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await progress_task

            while not self.progress_queue.empty():
                try:
                    remaining = self.progress_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                yield json.dumps(remaining, ensure_ascii=False)

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            custom_logger.error(f"Error in process_query: {e}")
            custom_logger.error(error_trace)
            
            error_content = f"처리 중 오류가 발생했습니다. 다시 시도해주세요."
            error_metadata = {}
            
            # 에러 응답을 스트리밍으로 전송 (HTTPException 대신)
            error_response = {
                "step": "done",
                "message_id": str(uuid.uuid4()),
                "role": "assistant",
                "content": error_content,
                "metadata": error_metadata,
                "created_at": datetime.utcnow().isoformat(),
                "error": str(e) if not isinstance(e, GraphRecursionError) else None
            }
            yield json.dumps(error_response, ensure_ascii=False)

    def _parse_plain_text_response(self, raw_content: str) -> dict[str, Any]:
        parsed_id = str(uuid.uuid4())
        parsed_content = raw_content
        parsed_metadata: dict[str, Any] = {}
        try:
            data = json.loads(raw_content)
            key_map = {
                "message_id": ["message_id", "message_id (필수 키값)"],
                "content": ["content", "content (필수 키값)"],
                "metadata": ["metadata", "metadata (필수 키값)"],
            }
            for target, candidates in key_map.items():
                for key in candidates:
                    if key in data:
                        if target == "message_id":
                            parsed_id = str(data[key])
                        elif target == "content":
                            parsed_content = str(data[key])
                        elif target == "metadata":
                            parsed_metadata = data[key] if isinstance(data[key], dict) else {}
                        break
        except (json.JSONDecodeError, TypeError):
            pass
        return {
            "step": "done",
            "message_id": parsed_id,
            "role": "assistant",
            "content": parsed_content,
            "metadata": self._handle_metadata(parsed_metadata),
            "created_at": datetime.utcnow().isoformat(),
        }

    @log_execution
    def _handle_metadata(self, metadata: dict[str, object] | None) -> dict[str, object]:
        custom_logger.info("========================================")
        custom_logger.info(metadata)
        result: dict[str, object] = {}
        if metadata:
            for k, v in metadata.items():
                result[k] = v
        return result
