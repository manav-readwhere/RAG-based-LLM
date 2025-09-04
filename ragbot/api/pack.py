import json
from typing import Any, Dict, List


def _compact(obj: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in obj.items() if v not in (None, "", [], {}, False) or v is True}


def _title_for(doc: Dict[str, Any]) -> str:
    table = doc.get("table") or doc.get("_table")
    if table == "users":
        name = " ".join([str(doc.get("first_name") or "").strip(), str(doc.get("last_name") or "").strip()]).strip()
        return f"user:{doc.get('id')}{(' '+name) if name else ''}".strip()
    if table == "campaigns":
        sc = doc.get("series_code") or (doc.get("campaign") or {}).get("series_code")
        return f"campaign:{sc or doc.get('id')}"
    if table == "sales_transactions":
        sc = (doc.get("campaign") or {}).get("series_code")
        return f"transaction:{doc.get('id')} campaign:{sc}" if sc else f"transaction:{doc.get('id')}"
    return doc.get("title") or "document"


def _content_for(doc: Dict[str, Any]) -> str:
    table = doc.get("table") or doc.get("_table")
    if table == "users":
        data = _compact({
            "type": "user",
            "id": doc.get("id"),
            "first_name": doc.get("first_name"),
            "last_name": doc.get("last_name"),
            "nationality": doc.get("nationality"),
            "date_of_birth": doc.get("date_of_birth"),
            "residency": _compact({
                "country": doc.get("country_of_residency"),
                "city": doc.get("city_of_residency"),
            }),
        })
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    if table == "campaigns":
        data = _compact({
            "type": "campaign",
            "id": doc.get("id"),
            "series_code": doc.get("series_code"),
            "campaign_type": doc.get("campaign_type"),
            "ticket_price": doc.get("ticket_price"),
            "start_date": doc.get("start_date"),
            "end_date": doc.get("end_date"),
            "total_tickets": doc.get("total_tickets"),
            "price_value": doc.get("price_value"),
        })
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    if table == "sales_transactions":
        user = doc.get("user") or {}
        campaign = doc.get("campaign") or {}
        data = _compact({
            "type": "transaction",
            "id": doc.get("id"),
            "transaction_id": doc.get("transaction_id"),
            "purchase_date_time": doc.get("purchase_date_time"),
            "payment_channel": doc.get("payment_channel"),
            "payment_method": doc.get("payment_method"),
            "currency": doc.get("currency"),
            "revenue": doc.get("revenue"),
            "is_first_time_buyer": doc.get("is_first_time_buyer"),
            "user": _compact({
                "id": user.get("id"),
                "nationality": user.get("nationality"),
                "country_of_residency": user.get("country_of_residency"),
                "city_of_residency": user.get("city_of_residency"),
            }),
            "campaign": _compact({
                "id": campaign.get("id"),
                "series_code": campaign.get("series_code"),
                "campaign_type": campaign.get("campaign_type"),
            }),
        })
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    # fallback: compact the existing content
    content = (doc.get("content") or "").strip()
    if len(content) > 1000:
        content = content[:1000]
    data = {"type": "document", "content": content}
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def compact_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compact: List[Dict[str, Any]] = []
    for d in docs:
        compact.append({
            "title": _title_for(d),
            "source": d.get("source", "es"),
            "url": d.get("url", ""),
            "content": _content_for(d),
        })
    return compact


