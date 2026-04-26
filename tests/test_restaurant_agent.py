from __future__ import annotations

import json
import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient


def collect_sse_events(response_text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in response_text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        try:
            events.append(json.loads(payload))
        except json.JSONDecodeError:
            pass
    return events


def get_done_event(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    for e in events:
        if e.get("step") == "done":
            return e
    return None


def chat(client: TestClient, message: str, thread_id: str | None = None) -> list[dict[str, Any]]:
    tid = thread_id or str(uuid.uuid4())
    resp = client.post(
        "/api/v1/chat",
        json={"thread_id": tid, "message": message},
    )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:300]}"
    return collect_sse_events(resp.text)


@pytest.fixture(scope="module")
def client():
    from app.main import app
    return TestClient(app)


@pytest.fixture
def tid():
    return str(uuid.uuid4())


@pytest.mark.order(10)
def test_root_endpoint(client: TestClient):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json().get("message") == "Edu Agent API"


@pytest.mark.order(11)
def test_health_endpoint(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json().get("status") == "healthy"


@pytest.mark.order(12)
def test_chat_returns_event_stream(client: TestClient, tid):
    resp = client.post(
        "/api/v1/chat",
        json={"thread_id": tid, "message": "안녕하세요"},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")


@pytest.mark.order(13)
def test_sse_contains_done_event(client: TestClient, tid):
    events = chat(client, "안녕하세요", tid)
    assert get_done_event(events) is not None, f"No 'done' event in: {events}"


@pytest.mark.order(14)
def test_done_event_has_required_fields(client: TestClient, tid):
    events = chat(client, "안녕하세요", tid)
    done = get_done_event(events)
    assert done is not None
    for field in ("message_id", "content", "metadata", "created_at"):
        assert field in done, f"Missing '{field}' in done event: {done}"


@pytest.mark.order(15)
def test_agent_responds_in_korean(client: TestClient, tid):
    events = chat(client, "안녕하세요! 맛집 추천해 주세요.", tid)
    done = get_done_event(events)
    assert done is not None
    content: str = done.get("content", "")
    has_korean = any("\uAC00" <= ch <= "\uD7A3" for ch in content)
    assert has_korean, f"Response does not appear to be Korean: {content[:200]}"


@pytest.mark.order(16)
def test_agent_greets_with_nonempty_content(client: TestClient, tid):
    events = chat(client, "안녕하세요", tid)
    done = get_done_event(events)
    assert done is not None
    assert len(done.get("content", "")) > 0


@pytest.mark.order(17)
def test_location_query_handled_gracefully(client: TestClient, tid):
    events = chat(client, "강남구에서 한식 맛집 추천해줘", tid)
    done = get_done_event(events)
    assert done is not None, "No final done event"
    assert isinstance(done.get("content"), str) and len(done["content"]) > 0


@pytest.mark.order(18)
def test_gps_query_handled_gracefully(client: TestClient, tid):
    events = chat(
        client,
        "위도 37.5172, 경도 127.0473 근처 1km 이내 식당 알려줘",
        tid,
    )
    done = get_done_event(events)
    assert done is not None
    assert isinstance(done.get("content"), str) and len(done["content"]) > 0


@pytest.mark.order(19)
def test_restaurant_id_query_handled_gracefully(client: TestClient, tid):
    events = chat(client, "식당 ID 'rest_001'의 상세 정보 알려줘", tid)
    done = get_done_event(events)
    assert done is not None
    assert done.get("content") is not None


@pytest.mark.order(20)
def test_es_unavailable_produces_meaningful_response(client: TestClient, tid):
    events = chat(client, "홍대에서 일식 식당 알려줘", tid)
    done = get_done_event(events)
    assert done is not None
    assert isinstance(done.get("content"), str) and len(done["content"]) > 0


@pytest.mark.order(21)
def test_english_query_handled(client: TestClient, tid):
    events = chat(client, "Can you recommend Korean BBQ restaurants in Gangnam?", tid)
    done = get_done_event(events)
    assert done is not None
    assert len(done.get("content", "")) > 0


@pytest.mark.order(22)
def test_multi_turn_context_retained(client: TestClient):
    tid = str(uuid.uuid4())

    events1 = chat(client, "이태원에서 맛집 추천해줘", tid)
    assert get_done_event(events1) is not None

    events2 = chat(client, "거기서 분위기 좋은 곳 있어?", tid)
    done2 = get_done_event(events2)
    assert done2 is not None
    assert len(done2.get("content", "")) > 0


@pytest.mark.order(23)
def test_initial_planning_step_present(client: TestClient, tid):
    resp = client.post(
        "/api/v1/chat",
        json={"thread_id": tid, "message": "강남 맛집 추천"},
    )
    assert resp.status_code == 200
    raw_lines = [
        line[6:] for line in resp.text.splitlines() if line.startswith("data: ")
    ]
    assert len(raw_lines) >= 1
    first_event = json.loads(raw_lines[0])
    assert first_event.get("step") == "model"
    assert "Planning" in first_event.get("tool_calls", [])


@pytest.mark.order(24)
def test_tools_step_has_name_field_when_present(client: TestClient, tid):
    events = chat(client, "강남구 한식 식당 검색해줘", tid)
    for te in (e for e in events if e.get("step") == "tools"):
        assert "name" in te, f"tools event missing 'name': {te}"
    assert get_done_event(events) is not None


@pytest.mark.order(25)
def test_done_event_message_id_is_valid_uuid(client: TestClient, tid):
    events = chat(client, "안녕하세요", tid)
    done = get_done_event(events)
    assert done is not None
    try:
        uuid.UUID(str(done.get("message_id", "")))
    except ValueError:
        pytest.fail(f"message_id is not a valid UUID: {done.get('message_id')!r}")


@pytest.mark.order(26)
def test_done_event_metadata_is_dict(client: TestClient, tid):
    events = chat(client, "안녕하세요", tid)
    done = get_done_event(events)
    assert done is not None
    assert isinstance(done.get("metadata"), dict), (
        f"metadata is not a dict: {done.get('metadata')!r}"
    )
