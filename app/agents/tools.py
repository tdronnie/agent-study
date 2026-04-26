# pyright: reportUnannotatedClassAttribute=false, reportAny=false, reportExplicitAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnusedCallResult=false, reportUnusedFunction=false
"""
Restaurant search tools for the LangGraph-based agent.

Dependencies required (add to pyproject.toml if not present):
    elasticsearch>=8.0.0
    langchain-elasticsearch>=0.3.0

Environment variables:
    ELASTICSEARCH_URL      : Elasticsearch endpoint (default: http://localhost:9200)
    ELASTICSEARCH_INDEX    : Index name for restaurant data (default: restaurants)
    ELASTICSEARCH_USERNAME : Basic-auth username (optional)
    ELASTICSEARCH_PASSWORD : Basic-auth password (optional)
"""

from __future__ import annotations

import logging
from urllib.parse import unquote
from typing import Any

import httpx

from app.core.config import settings
from langchain_core.tools import tool

logger = logging.getLogger("edu_agent")

_ES_URL: str = settings.ELASTICSEARCH_URL
_ES_INDEX: str = settings.ELASTICSEARCH_INDEX
_ES_API_KEY: str = settings.ELASTICSEARCH_API_KEY
_ES_USERNAME: str | None = settings.ELASTICSEARCH_USERNAME
_ES_PASSWORD: str | None = settings.ELASTICSEARCH_PASSWORD
_REST_TIMEOUT_SECONDS: float = 10.0


def _call_rest_api(params: dict[str, Any]) -> dict[str, Any]:
    """Call the restaurant OpenAPI and return the parsed JSON payload."""
    if not settings.REST_URL:
        raise RuntimeError("Restaurant OpenAPI URL이 설정되지 않았습니다.")

    if not settings.REST_API_KEY:
        raise RuntimeError("Restaurant OpenAPI 키가 설정되지 않았습니다.")

    url = f"{settings.REST_URL.rstrip('/')}/info"
    api_keys = [settings.REST_API_KEY]
    decoded_key = unquote(settings.REST_API_KEY)
    if decoded_key and decoded_key != settings.REST_API_KEY:
        api_keys.append(decoded_key)

    last_error: Exception | None = None

    for api_key in api_keys:
        request_params = {
            "serviceKey": api_key,
            "type": "json",
            **params,
        }

        try:
            response = httpx.get(url, params=request_params, timeout=_REST_TIMEOUT_SECONDS)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError("Restaurant OpenAPI 응답 형식이 올바르지 않습니다.")
            return payload
        except httpx.TimeoutException as exc:
            last_error = exc
        except httpx.HTTPStatusError as exc:
            last_error = exc
        except httpx.RequestError as exc:
            last_error = exc
        except ValueError as exc:
            last_error = exc

    raise RuntimeError(f"Restaurant OpenAPI 호출에 실패했습니다: {last_error}")


def _extract_rest_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract restaurant items from common OpenAPI response shapes."""
    def _nested_get(value: Any, *keys: str) -> Any:
        current = value
        for key in keys:
            if not isinstance(current, dict):
                return None
            current = current.get(key)
        return current

    candidates: list[Any] = [
        _nested_get(payload, "response", "body", "items", "item"),
        _nested_get(payload, "body", "items", "item"),
        _nested_get(payload, "items", "item"),
        payload.get("item"),
        payload.get("data"),
    ]

    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, list):
            return [item for item in candidate if isinstance(item, dict)]
        if isinstance(candidate, dict):
            return [candidate]

    return []


def _rest_item_to_dict(item: dict[str, Any]) -> dict[str, Any]:
    """Map restaurant OpenAPI fields to the agent's output schema."""
    return {
        "id": item.get("MNG_NO", ""),
        "name": item.get("BPLC_NM", ""),
        "address": item.get("ROAD_NM_ADDR") or item.get("LOTNO_ADDR", ""),
        "phone": item.get("TELNO", ""),
        "cuisine_type": item.get("BZSTAT_SE_NM", ""),
        "status": item.get("SALS_STTS_NM", ""),
    }

def _get_es_client() -> Any | None:
    try:
        from elasticsearch import Elasticsearch  # type: ignore[import]
    except ImportError:
        logger.error("elasticsearch 패키지가 설치되지 않았습니다. 'uv add elasticsearch' 또는 'pip install elasticsearch'를 실행하세요.")
        return None

    try:
        if _ES_API_KEY:
            client = Elasticsearch(_ES_URL, api_key=_ES_API_KEY)
        elif _ES_USERNAME and _ES_PASSWORD:
            client = Elasticsearch(_ES_URL, basic_auth=(_ES_USERNAME, _ES_PASSWORD))
        else:
            client = Elasticsearch(_ES_URL)

        if not client.ping():
            logger.warning("Elasticsearch 서버에 연결할 수 없습니다: %s", _ES_URL)
            return None

        return client

    except Exception as exc:
        logger.error("Elasticsearch 클라이언트 초기화 실패: %s", exc)
        return None


def _get_es_retriever(query_fn: Any | None = None) -> Any | None:
    try:
        from langchain_elasticsearch import ElasticsearchRetriever  # type: ignore[import]
    except ImportError:
        logger.error("langchain-elasticsearch 패키지가 설치되지 않았습니다. 'uv add langchain-elasticsearch'를 실행하세요.")
        return None

    try:
        def _default_query(search_query: str) -> dict[str, Any]:
            return {
                "query": {
                    "multi_match": {
                        "query": search_query,
                        "fields": ["name^2", "description", "cuisine_type", "location"],
                    }
                },
                "size": 10,
            }

        effective_query_fn = query_fn or _default_query

        client = _get_es_client()
        if client is None:
            return None

        return ElasticsearchRetriever(
            client=client,
            index_name=_ES_INDEX,
            content_field="description",
            body_func=effective_query_fn,
        )

    except Exception as exc:
        logger.error("ElasticsearchRetriever 초기화 실패: %s", exc)
        return None


def _hit_to_dict(hit: dict[str, Any], include_distance: bool = False) -> dict[str, Any]:
    source: dict[str, Any] = hit.get("_source", {})
    result: dict[str, Any] = {
        "id": source.get("store_id", hit.get("_id", "")),
        "name": source.get("menu_name_ko", ""),
        "location": source.get("store_region", ""),
        "cuisine_type": source.get("store_type", ""),
        "price": source.get("price", ""),
    }
    if include_distance:
        sort_values = hit.get("sort", [])
        result["distance_km"] = round(sort_values[0], 2) if sort_values else None
    return result


@tool
def search_restaurants(
    location: str,
    cuisine_type: str | None = None,
) -> list[dict[str, Any]]:
    """지역과 음식 종류로 식당을 검색합니다.

    Elasticsearch에서 store_region 필드와 선택적으로 store_type 필드를 기준으로
    식당 목록을 조회합니다.

    Args:
        location: 검색할 지역/구역 (예: "강남구", "홍대", "이태원")
        cuisine_type: 필터링할 음식 종류 (예: "한식", "일식", "양식", "중식").
                      지정하지 않으면 모든 음식 종류를 반환합니다.

    Returns:
        식당 정보 딕셔너리 목록. 각 항목은 id, name, location,
        cuisine_type, price 키를 포함합니다.
        오류 발생 시 error 키를 포함한 딕셔너리를 반환합니다.
    """
    client = _get_es_client()
    if client is None:
        return [{"error": "Elasticsearch에 연결할 수 없습니다."}]

    try:
        must_clauses: list[dict[str, Any]] = [{"match": {"store_region": location}}]
        if cuisine_type:
            must_clauses.append({"match": {"store_type": cuisine_type}})

        query: dict[str, Any] = {
            "query": {"bool": {"must": must_clauses}},
            "size": 10,
        }

        response = client.search(index=_ES_INDEX, body=query)
        hits: list[dict[str, Any]] = response.get("hits", {}).get("hits", [])

        if not hits:
            return [{"message": f"'{location}'에서 식당을 찾을 수 없습니다."}]

        return [_hit_to_dict(hit) for hit in hits]

    except Exception as exc:
        logger.error("search_restaurants 오류: %s", exc)
        return [{"error": f"검색 중 오류가 발생했습니다: {exc}"}]


@tool
def get_restaurant_info(restaurant_id: str) -> dict[str, Any]:
    """특정 식당의 상세 정보를 OpenAPI로 조회합니다.

    restaurant_id를 MNG_NO 조건으로 사용하여 식당 1건의 상세 정보를 조회합니다.

    Args:
        restaurant_id: 식당의 고유 식별자 (MNG_NO)

    Returns:
        식당 상세 정보 딕셔너리. id, name, address, phone, cuisine_type,
        status 키를 포함합니다. 식당을 찾을 수 없거나 오류 발생 시 error 키를 포함합니다.
    """
    try:
        payload = _call_rest_api({"cond[MNG_NO::EQ]": restaurant_id})
        items = _extract_rest_items(payload)

        if not items:
            return {"error": f"ID '{restaurant_id}'에 해당하는 식당을 찾을 수 없습니다."}

        return _rest_item_to_dict(items[0])

    except Exception as exc:
        logger.error("get_restaurant_info 오류 (id=%s): %s", restaurant_id, exc)
        return {"error": f"식당 정보 조회 중 오류가 발생했습니다: {exc}"}


@tool
def find_nearby_restaurants(
    location: str,
    cuisine_type: str | None = None,
) -> list[dict[str, Any]]:
    """지역명으로 주변 식당을 OpenAPI에서 검색합니다.

    ROAD_NM_ADDR 조건으로 location을 검색하고, cuisine_type이 주어지면
    BZSTAT_SE_NM 조건을 추가로 사용합니다.

    Args:
        location: 검색할 지역/도로명 주소 일부
        cuisine_type: 필터링할 음식 종류 (예: "한식", "일식"). 지정하지 않으면 모든 음식 종류를 반환합니다.

    Returns:
        식당 정보 딕셔너리 목록. 각 항목은 id, name, address, phone,
        cuisine_type, status 키를 포함합니다. 오류 발생 시 error 키를 포함한 딕셔너리를 반환합니다.
    """
    try:
        params: dict[str, Any] = {"cond[ROAD_NM_ADDR::LIKE]": location}
        if cuisine_type:
            params["cond[BZSTAT_SE_NM::LIKE]"] = cuisine_type

        payload = _call_rest_api(params)
        items = _extract_rest_items(payload)

        if not items:
            return [{"message": f"'{location}'에서 식당을 찾을 수 없습니다."}]

        return [_rest_item_to_dict(item) for item in items]

    except Exception as exc:
        logger.error("find_nearby_restaurants 오류 (location=%s): %s", location, exc)
        return [{"error": f"근처 식당 검색 중 오류가 발생했습니다: {exc}"}]


restaurant_tools = [
    search_restaurants,
    get_restaurant_info,
    find_nearby_restaurants,
]
