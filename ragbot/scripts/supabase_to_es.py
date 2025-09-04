import argparse
import os
import json
import time
import logging
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Iterable

from dotenv import load_dotenv
from supabase import create_client, Client

from ragbot.api.embeddings import embed_many
from ragbot.api.env import embedding_dimensions
from ragbot.api.elastic import get_client


logger = logging.getLogger("ragbot.migrate")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


# ---------- Supabase helpers ----------

def get_supabase_client() -> Client:
    load_dotenv(override=False)
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_KEY", "")
    if not url or not key:
        raise RuntimeError(
            "Missing SUPABASE_URL or SUPABASE_KEY. Export SUPABASE_URL and SUPABASE_KEY (service role key recommended)."
        )
    host = url.split("//")[-1].split("/")[0]
    logger.info("Connecting to Supabase host=%s", host)
    return create_client(url, key)


def fetch_table_rows(sb: Client, table: str, limit: Optional[int], fetch_batch_size: int) -> Iterable[List[Dict]]:
    logger.info("Fetch table rows start table=%s limit=%s batch_size=%s", table, limit, fetch_batch_size)
    offset = 0
    total = 0
    while True:
        page_size = fetch_batch_size if limit is None else min(fetch_batch_size, max(0, limit - total))
        if page_size == 0:
            break
        t0 = time.time()
        res = sb.table(table).select("*").range(offset, offset + page_size - 1).execute()
        dt = int((time.time() - t0) * 1000)
        rows = res.data or []  # type: ignore[attr-defined]
        logger.info("Fetched rows table=%s offset=%s count=%s dt_ms=%s", table, offset, len(rows), dt)
        if not rows:
            break
        yield rows
        read_n = len(rows)
        total += read_n
        offset += read_n
    logger.info("Fetch table rows end table=%s total_rows=%s", table, total)


def fetch_by_ids(sb: Client, table: str, id_field: str, ids: List[str], batch_size: int = 1000) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for i in range(0, len(ids), batch_size):
        chunk = list({x for x in ids[i : i + batch_size] if x})
        if not chunk:
            continue
        t0 = time.time()
        res = sb.table(table).select("*").in_(id_field, chunk).execute()
        dt = int((time.time() - t0) * 1000)
        rows = (res.data or [])  # type: ignore[attr-defined]
        for r in rows:
            out[str(r.get("id"))] = r
        logger.info("Fetched %s ids from %s in %sms", len(rows), table, dt)
    return out


# ---------- OpenAI embedding batching ----------

def embed_batched(texts: List[str], batch_size: int, max_retries: int = 3, base_delay: float = 1.5) -> List[List[float]]:
    vectors: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        attempt = 0
        while True:
            try:
                t0 = time.time()
                logger.info("Embedding batch start idx=%s size=%s attempt=%s", i, len(batch), attempt + 1)
                part = embed_many(batch)
                dt = int((time.time() - t0) * 1000)
                logger.info("Embedding batch done idx=%s size=%s dt_ms=%s", i, len(batch), dt)
                vectors.extend(part)
                break
            except Exception as e:
                attempt += 1
                logger.warning("Embedding batch failed idx=%s attempt=%s error=%s", i, attempt, e)
                if attempt >= max_retries:
                    logger.error("Embedding batch giving up idx=%s attempts=%s", i, attempt)
                    raise
                sleep_s = base_delay * (2 ** (attempt - 1))
                logger.info("Retrying embedding batch idx=%s in %.2fs", i, sleep_s)
                time.sleep(sleep_s)
    return vectors


# ---------- Elasticsearch helpers ----------

def ensure_index_with_mappings(index_name: str, mappings: Dict) -> None:
    es = get_client()
    if es.indices.exists(index=index_name):
        logger.info("Index exists index=%s", index_name)
        return
    es.indices.create(index=index_name, mappings=mappings)
    logger.info("Index created index=%s", index_name)


def users_mappings(dims: int) -> Dict:
    return {
        "properties": {
            "id": {"type": "keyword"},
            "table": {"type": "keyword"},
            "source": {"type": "keyword"},
            "created_at": {"type": "date"},
            "content": {"type": "text"},
            "embedding": {"type": "dense_vector", "dims": dims, "index": True, "similarity": "cosine"},
            # user fields
            "first_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "last_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "email": {"type": "keyword"},
            "mobile_number": {"type": "keyword"},
            "date_of_birth": {"type": "date"},
            "nationality": {"type": "keyword"},
            "id_type": {"type": "keyword"},
            "id_number": {"type": "keyword"},
            "country_of_residency": {"type": "keyword"},
            "city_of_residency": {"type": "keyword"},
        }
    }


def campaigns_mappings(dims: int) -> Dict:
    return {
        "properties": {
            "id": {"type": "keyword"},
            "table": {"type": "keyword"},
            "source": {"type": "keyword"},
            "created_at": {"type": "date"},
            "content": {"type": "text"},
            "embedding": {"type": "dense_vector", "dims": dims, "index": True, "similarity": "cosine"},
            # campaign fields
            "series_code": {"type": "keyword"},
            "campaign_type": {"type": "keyword"},
            "ticket_price": {"type": "scaled_float", "scaling_factor": 100},
            "start_date": {"type": "date"},
            "end_date": {"type": "date"},
            "total_tickets": {"type": "integer"},
            "price_value": {"type": "keyword"},
        }
    }


def sales_mappings(dims: int) -> Dict:
    return {
        "properties": {
            "id": {"type": "keyword"},
            "table": {"type": "keyword"},
            "source": {"type": "keyword"},
            "created_at": {"type": "date"},
            "content": {"type": "text"},
            "embedding": {"type": "dense_vector", "dims": dims, "index": True, "similarity": "cosine"},
            # flat sales fields
            "transaction_id": {"type": "keyword"},
            "ticket_number": {"type": "keyword"},
            "purchase_date": {"type": "date"},
            "purchase_time": {"type": "keyword"},
            "purchase_date_time": {"type": "date"},
            "payment_method": {"type": "keyword"},
            "payment_channel": {"type": "keyword"},
            "price_segment": {"type": "keyword"},
            "currency": {"type": "keyword"},
            "revenue": {"type": "scaled_float", "scaling_factor": 100},
            "is_first_time_buyer": {"type": "boolean"},
            "customer_segment": {"type": "keyword"},
            "is_weekend": {"type": "boolean"},
            "user_id": {"type": "keyword"},
            "campaign_id": {"type": "keyword"},
            # nested/denormalized objects
            "user": {
                "type": "object",
                "properties": {
                    "id": {"type": "keyword"},
                    "first_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                    "last_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                    "email": {"type": "keyword"},
                    "nationality": {"type": "keyword"},
                    "country_of_residency": {"type": "keyword"},
                    "city_of_residency": {"type": "keyword"},
                },
            },
            "campaign": {
                "type": "object",
                "properties": {
                    "id": {"type": "keyword"},
                    "series_code": {"type": "keyword"},
                    "campaign_type": {"type": "keyword"},
                    "ticket_price": {"type": "scaled_float", "scaling_factor": 100},
                    "start_date": {"type": "date"},
                    "end_date": {"type": "date"},
                },
            },
        }
    }


def bulk_index(index_name: str, docs: List[Dict]) -> Tuple[int, int]:
    from elasticsearch.helpers import bulk

    es = get_client()
    actions = (
        {"_op_type": "index", "_index": index_name, "_id": d.get("id"), "_source": d} for d in docs
    )
    t0 = time.time()
    success, _ = bulk(es, actions)
    dt = int((time.time() - t0) * 1000)
    logger.info("Bulk indexed index=%s docs=%s success=%s dt_ms=%s", index_name, len(docs), success, dt)
    return int(success), len(docs)


# ---------- Content builders ----------

def dict_to_lines(d: Dict) -> List[str]:
    lines: List[str] = []
    for k, v in d.items():
        try:
            val = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
        except Exception:
            val = str(v)
        lines.append(f"- {k}: {val}")
    return lines


def to_text(table: str, row: Dict) -> str:
    parts = [f"Table: {table}", "Columns:"]
    parts.extend(dict_to_lines(row))
    return "\n".join(parts)


def summarize_user(u: Dict) -> str:
    keys = [
        "first_name",
        "last_name",
        "email",
        "mobile_number",
        "nationality",
        "date_of_birth",
        "country_of_residency",
        "city_of_residency",
    ]
    info = {k: u.get(k) for k in keys if k in u}
    return "\n".join(["User Summary:"] + dict_to_lines(info))


def summarize_campaign(c: Dict) -> str:
    keys = [
        "series_code", "campaign_type", "ticket_price", "start_date", "end_date", "total_tickets", "prize_value",
    ]
    info = {k: c.get(k) for k in keys if k in c}
    return "\n".join(["Campaign Summary:"] + dict_to_lines(info))


def build_user_doc(row: Dict, now_iso: str, embedding: List[float]) -> Dict:
    return {
        "id": str(row.get("id")),
        "table": "users",
        "source": "supabase",
        "content": to_text("users", row),
        "embedding": embedding,
        "created_at": now_iso,
        # flattened user fields (for filters/aggregations)
        "first_name": row.get("first_name"),
        "last_name": row.get("last_name"),
        "email": row.get("email"),
        "mobile_number": row.get("mobile_number"),
        "date_of_birth": row.get("date_of_birth"),
        "nationality": row.get("nationality"),
        "id_type": row.get("id_type"),
        "id_number": row.get("id_number"),
        "country_of_residency": row.get("country_of_residency"),
        "city_of_residency": row.get("city_of_residency"),
    }


def build_campaign_doc(row: Dict, now_iso: str, embedding: List[float]) -> Dict:
    return {
        "id": str(row.get("id")),
        "table": "campaigns",
        "source": "supabase",
        "content": to_text("campaigns", row),
        "embedding": embedding,
        "created_at": now_iso,
        # flattened campaign fields
        "series_code": row.get("series_code"),
        "campaign_type": row.get("campaign_type"),
        "ticket_price": row.get("ticket_price"),
        "start_date": row.get("start_date"),
        "end_date": row.get("end_date"),
        "total_tickets": row.get("total_tickets"),
        "prize_value": row.get("prize_value"),
    }


def build_sale_doc(row: Dict, now_iso: str, embedding: List[float], users_by_id: Dict[str, Dict], campaigns_by_id: Dict[str, Dict]) -> Dict:
    user_id = str(row.get("user_id")) if row.get("user_id") is not None else None
    campaign_id = str(row.get("campaign_id")) if row.get("campaign_id") is not None else None
    parts = [to_text("sales_transactions", row)]
    if user_id and user_id in users_by_id:
        parts.append(summarize_user(users_by_id[user_id]))
    if campaign_id and campaign_id in campaigns_by_id:
        parts.append(summarize_campaign(campaigns_by_id[campaign_id]))
    content = "\n\n".join(parts)
    user_obj = users_by_id.get(user_id, {}) if user_id else {}
    campaign_obj = campaigns_by_id.get(campaign_id, {}) if campaign_id else {}
    return {
        "id": str(row.get("id")),
        "table": "sales_transactions",
        "source": "supabase",
        "content": content,
        "embedding": embedding,
        "created_at": now_iso,
        # fk
        "user_id": user_id,
        "campaign_id": campaign_id,
        # flat sales fields
        "transaction_id": row.get("transaction_id"),
        "ticket_number": row.get("ticket_number"),
        "purchase_date": row.get("purchase_date"),
        "purchase_time": row.get("purchase_time"),
        "purchase_date_time": row.get("purchase_date_time") or row.get("purchase_date"),
        "payment_method": row.get("payment_method"),
        "payment_channel": row.get("payment_channel"),
        "price_segment": row.get("price_segment"),
        "currency": row.get("currency"),
        "revenue": row.get("revenue"),
        "is_first_time_buyer": row.get("is_first_time_buyer"),
        "customer_segment": row.get("customer_segment"),
        "is_weekend": row.get("is_weekend"),
        # denormalized objects
        "user": {
            "id": user_obj.get("id"),
            "first_name": user_obj.get("first_name"),
            "last_name": user_obj.get("last_name"),
            "email": user_obj.get("email"),
            "nationality": user_obj.get("nationality"),
            "country_of_residency": user_obj.get("country_of_residency"),
            "city_of_residency": user_obj.get("city_of_residency"),
        } if user_obj else None,
        "campaign": {
            "id": campaign_obj.get("id"),
            "series_code": campaign_obj.get("series_code"),
            "campaign_type": campaign_obj.get("campaign_type"),
            "ticket_price": campaign_obj.get("ticket_price"),
            "start_date": campaign_obj.get("start_date"),
            "end_date": campaign_obj.get("end_date"),
        } if campaign_obj else None,
    }


# ---------- Migration (streaming) ----------

def migrate_simple_stream(sb: Client, table: str, index_name: str, dims: int, doc_builder, embed_batch_size: int, fetch_batch_size: int, index_batch_size: int, limit: Optional[int]) -> None:
    logger.info("Migrate start table=%s index=%s", table, index_name)
    # choose mappings per table
    if table == "users":
        ensure_index_with_mappings(index_name, users_mappings(dims))
    elif table == "campaigns":
        ensure_index_with_mappings(index_name, campaigns_mappings(dims))
    else:
        ensure_index_with_mappings(index_name, {"properties": {}})
    total_docs = 0
    for batch in fetch_table_rows(sb, table=table, limit=limit, fetch_batch_size=fetch_batch_size):
        texts = [to_text(table, r) for r in batch]
        vectors = embed_batched(texts, batch_size=embed_batch_size)
        now = datetime.now(timezone.utc).isoformat()
        docs = [doc_builder(row, now, vec) for row, vec in zip(batch, vectors)]
        # index in smaller chunks to keep memory low
        for i in range(0, len(docs), index_batch_size):
            chunk = docs[i : i + index_batch_size]
            bulk_index(index_name, chunk)
        total_docs += len(docs)
        logger.info("Migrate batch indexed table=%s batch=%s total=%s", table, len(docs), total_docs)
    logger.info("Migrate done table=%s total_indexed=%s", table, total_docs)


def migrate_sales_stream(sb: Client, index_name: str, dims: int, embed_batch_size: int, fetch_batch_size: int, index_batch_size: int, limit: Optional[int]) -> None:
    logger.info("Migrate sales start index=%s", index_name)
    ensure_index_with_mappings(index_name, sales_mappings(dims))
    total_docs = 0
    for sales_batch in fetch_table_rows(sb, table="sales_transactions", limit=limit, fetch_batch_size=fetch_batch_size):
        # fetch related users/campaigns only for this batch
        user_ids = [str(r.get("user_id")) for r in sales_batch if r.get("user_id") is not None]
        camp_ids = [str(r.get("campaign_id")) for r in sales_batch if r.get("campaign_id") is not None]
        users_by_id = fetch_by_ids(sb, "users", "id", user_ids)
        campaigns_by_id = fetch_by_ids(sb, "campaigns", "id", camp_ids)
        # build enriched texts
        texts: List[str] = []
        rows: List[Dict] = []
        for r in sales_batch:
            user_id = str(r.get("user_id")) if r.get("user_id") is not None else None
            campaign_id = str(r.get("campaign_id")) if r.get("campaign_id") is not None else None
            parts = [to_text("sales_transactions", r)]
            if user_id and user_id in users_by_id:
                parts.append(summarize_user(users_by_id[user_id]))
            if campaign_id and campaign_id in campaigns_by_id:
                parts.append(summarize_campaign(campaigns_by_id[campaign_id]))
            texts.append("\n\n".join(parts))
            rows.append(r)
        vectors = embed_batched(texts, batch_size=embed_batch_size)
        now = datetime.now(timezone.utc).isoformat()
        docs = [build_sale_doc(r, now, v, users_by_id, campaigns_by_id) for r, v in zip(rows, vectors)]
        for i in range(0, len(docs), index_batch_size):
            chunk = docs[i : i + index_batch_size]
            bulk_index(index_name, chunk)
        total_docs += len(docs)
        logger.info("Migrate sales batch indexed batch=%s total=%s", len(docs), total_docs)
    logger.info("Migrate sales done total_indexed=%s", total_docs)


def delete_indices(indices: List[str], alias: Optional[str] = None) -> None:
    es = get_client()
    # Remove alias if present
    if alias:
        try:
            es.indices.delete_alias(index="*", name=alias)
            logger.info("Deleted alias name=%s", alias)
        except Exception:
            logger.info("Alias not found or already removed name=%s", alias)
    # Delete indices
    for idx in indices:
        try:
            if es.indices.exists(index=idx):
                es.indices.delete(index=idx)
                logger.info("Deleted index index=%s", idx)
            else:
                logger.info("Index not found index=%s", idx)
        except Exception as e:
            logger.warning("Failed deleting index=%s error=%s", idx, e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Streamed Supabase→Elasticsearch migration (memory-optimized)")
    parser.add_argument("--users-index", default="users", help="Elasticsearch index name for users")
    parser.add_argument("--campaigns-index", default="campaigns", help="Elasticsearch index name for campaigns")
    parser.add_argument("--sales-index", default="sales_transactions", help="Elasticsearch index name for sales")
    parser.add_argument("--alias", default="kb_docs", help="Alias to point chatbot retrieval at (default: kb_docs)")
    parser.add_argument("--reset", action="store_true", help="Delete existing indices and alias before migration")
    parser.add_argument("--embed-batch-size", type=int, default=500, help="Embedding batch size")
    parser.add_argument("--fetch-batch-size", type=int, default=500, help="Rows fetched per request")
    parser.add_argument("--index-batch-size", type=int, default=1000, help="Docs per bulk index chunk")
    parser.add_argument("--limit-per-table", type=int, default=None, help="Optional row limit per table")
    args = parser.parse_args()

    logger.info(
        "Migration start users_index=%s campaigns_index=%s sales_index=%s alias=%s fetch_batch=%s embed_batch=%s index_batch=%s",
        args.users_index, args.campaigns_index, args.sales_index, args.alias, args.fetch_batch_size, args.embed_batch_size, args.index_batch_size,
    )

    sb = get_supabase_client()
    dims = embedding_dimensions(os.getenv("EMBED_MODEL", "text-embedding-3-large"))
    logger.info("Embedding dims=%s", dims)

    if args.reset:
        delete_indices([args.users_index, args.campaigns_index, args.sales_index], alias=args.alias)

    # Users and campaigns streamed
    migrate_simple_stream(sb, "users", args.users_index, dims, build_user_doc, args.embed_batch_size, args.fetch_batch_size, args.index_batch_size, args.limit_per_table)
    migrate_simple_stream(sb, "campaigns", args.campaigns_index, dims, build_campaign_doc, args.embed_batch_size, args.fetch_batch_size, args.index_batch_size, args.limit_per_table)

    # Sales with per-batch joins
    migrate_sales_stream(sb, args.sales_index, dims, args.embed_batch_size, args.fetch_batch_size, args.index_batch_size, args.limit_per_table)

    # Alias (point chatbot to all three indices)
    from elasticsearch import Elasticsearch
    es = get_client()
    actions = []
    try:
        current = es.indices.get_alias(name=args.alias)
        for idx in current.keys():
            if idx not in [args.users_index, args.campaigns_index, args.sales_index]:
                actions.append({"remove": {"index": idx, "alias": args.alias}})
    except Exception:
        pass
    for idx in [args.users_index, args.campaigns_index, args.sales_index]:
        actions.append({"add": {"index": idx, "alias": args.alias}})
    if actions:
        es.indices.update_aliases(body={"actions": actions})
    logger.info("Migration complete. Alias %s → %s", args.alias, [args.users_index, args.campaigns_index, args.sales_index])


if __name__ == "__main__":
    main()
