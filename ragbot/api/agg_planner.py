import json
import logging
import re
from typing import Any, Dict, Optional

from .openai_client import chat_completion
from .elastic import get_client


logger = logging.getLogger("ragbot.agg")


PLANNER_SYSTEM = """
You are an assistant that converts natural language questions into Elasticsearch queries.
You will be provided with:
1. A user query in plain language.
2. The skeleton of available Elasticsearch indices and their fields.

Your task is to return a valid Elasticsearch query in JSON format.
- Use `term` or `match` for exact lookups (e.g., series_code, email, nationality).
- Use `range` for date or numeric filtering (e.g., purchase_date, revenue).
- Use `aggs` for aggregation queries (e.g., sum, avg, count, group by).
- Use `size: 0` when returning only aggregated results.
- Return only the JSON query (no explanation).

Indices Schema:
1. sales_transactions
   Fields: 
     id, user_id, campaign_id, transaction_id, ticket_number, purchase_date, purchase_time, purchase_date_time,
     payment_method, payment_channel, price_segment, currency, revenue, is_first_time_buyer, customer_segment,
     created_at, is_weekend,
     user (object: id, first_name, last_name, email, nationality, country_of_residency, city_of_residency),
     campaign (object: id, series_code, campaign_type, ticket_price, start_date, end_date, total_tickets, prize_value)

2. campaigns
   Fields: 
     id, series_code, campaign_type, ticket_price, start_date, end_date, total_tickets, prize_value

3. users
   Fields: 
     id, first_name, last_name, email, mobile_number, date_of_birth, nationality, id_type, id_number,
     country_of_residency, city_of_residency

Examples:

User Query:
Provide me sales of FSC25093

Output:
{
  "size": 0,
  "query": {
    "term": {
      "campaign.series_code": "FSC25093"
    }
  },
  "aggs": {
    "total_sales": {
      "sum": {
        "field": "revenue"
      }
    }
  }
}

User Query:
List top 5 customers by revenue in September 2025

Output:
{
  "size": 0,
  "query": {
    "range": {
      "purchase_date": {
        "gte": "2025-09-01",
        "lte": "2025-09-30"
      }
    }
  },
  "aggs": {
    "top_customers": {
      "terms": {
        "field": "user.id",
        "size": 5,
        "order": { "total_spent": "desc" }
      },
      "aggs": {
        "total_spent": {
          "sum": { "field": "revenue" }
        }
      }
    }
  }
}

User Query:
Get average ticket price for campaign FSC25033

Output:
{
  "size": 0,
  "query": {
    "term": {
      "campaign.series_code": "FSC25033"
    }
  },
  "aggs": {
    "avg_ticket_price": {
      "avg": {
        "field": "campaign.ticket_price"
      }
    }
  }
}

User Query:
How many transactions happened on weekends?

Output:
{
  "size": 0,
  "query": {
    "term": {
      "is_weekend": true
    }
  }
}
"""


def _extract_json_object(s: str) -> Optional[str]:
    # Try fenced block first
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", s, re.IGNORECASE)
    if m:
        return m.group(1)
    # Find first balanced {...}
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return s[start:i+1]
    return None


def _parse_json(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    # direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # extract and parse
    extracted = _extract_json_object(text)
    if extracted:
        try:
            return json.loads(extracted)
        except Exception:
            return None
    return None


def plan_aggregation(query: str, rid: Optional[str] = None) -> Optional[Dict[str, Any]]:
    _q_preview = query[:120].replace("\n", " ")
    logger.info(f"[{rid or '-'}] agg plan start query='{_q_preview}'")
    messages = [
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": f"User question: {query}\n\nReturn Elasticsearch query only."},
    ]
    out_any = chat_completion(messages, stream=False)  # type: ignore[assignment]
    # Normalize output to a string even if a generator slips through
    out_str: str = ""
    if isinstance(out_any, str):
        out_str = out_any
    else:
        try:
            out_str = "".join(list(out_any))  # type: ignore[arg-type]
            logger.warning(f"[{rid or '-'}] agg plan received generator; concatenated to string of len={len(out_str)}")
        except Exception:
            logger.warning(f"[{rid or '-'}] agg plan non_string_output type={type(out_any)} and could not concatenate")
            out_str = ""
    if out_str:
        preview = out_str.replace("\n", " ")[:200]
        logger.info(f"[{rid or '-'}] agg plan raw_out_preview='{preview}'")
        logger.info(f"[{rid or '-'}] agg plan raw_out_full=\n{out_str}")
    body = _parse_json(out_str)
    if not body or not isinstance(body, dict):
        logger.warning(f"[{rid or '-'}] agg plan parse_failed")
        return None
    # enforce safe defaults
    body.setdefault("size", 0)
    if "aggs" not in body:
        logger.warning(f"[{rid or '-'}] agg plan missing_aggs")
        return None
    logger.info(f"[{rid or '-'}] agg plan ok keys={list(body.keys())}")
    return body

def run_aggregation(body: Dict[str, Any], rid: Optional[str] = None) -> Dict[str, Any]:
    logger.info(f"[{rid or '-'}] agg run size={body.get('size')} has_aggs={'aggs' in body}")
    es = get_client()
    resp = es.search(index="sales_transactions", body=body)
    return resp


def plan_and_run(query: str, rid: Optional[str] = None) -> Optional[Dict[str, Any]]:
    body = plan_aggregation(query, rid=rid)
    if not body:
        logger.info(f"[{rid or '-'}] agg plan_and_run no_body")
        return None
    try:
        resp = run_aggregation(body, rid=rid)
        aggs = resp.get("aggregations") or {}
        # log a compact summary of top-level agg values
        summary = {k: v.get("value") if isinstance(v, dict) and "value" in v else "obj" for k, v in aggs.items()}
        logger.info(f"[{rid or '-'}] agg run ok summary={summary}")
        return {"body": body, "aggregations": aggs}
    except Exception as e:
        logger.warning(f"[{rid or '-'}] agg run failed error={e}")
        return None


