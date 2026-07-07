"""Milvus 向量 RAG：本地 Embedding + 商品/帮助知识检索。"""

import logging
import re
from functools import lru_cache
from typing import Any
from bs4 import BeautifulSoup
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)
from sentence_transformers import SentenceTransformer
from sqlalchemy import text
from app.config.settings import get_settings
from app.models.db import MallSessionLocal

settings = get_settings()
logger = logging.getLogger(__name__)
DIM = 512  # bge-small-zh-v1.5 向量维度


@lru_cache
def get_embedder() -> SentenceTransformer:
    """获取本地 Embedding 模型单例。
    Returns:
        SentenceTransformer: 加载配置的 sentence-transformers 模型。
    """
    return SentenceTransformer(settings.embedding_model)


def _strip_html(html: str | None) -> str:
    """去除 HTML 标签，提取纯文本。
    Args:
        html: HTML 字符串，可为空。
    Returns:
        str: 清洗后的纯文本。
    """
    if not html:
        return ""
    return BeautifulSoup(html, "html.parser").get_text(separator=" ", strip=True)


def _connect_milvus() -> None:
    """建立 Milvus 连接（alias=default）。"""
    connections.connect(
        alias="default", host=settings.milvus_host, port=settings.milvus_port
    )


def ensure_collection() -> Collection:
    """确保 Milvus 集合存在并已建索引、加载到内存。
    Returns:
        Collection: 可搜索的知识库集合实例。
    """
    _connect_milvus()
    name = settings.milvus_collection
    if not utility.has_collection(name):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIM),
        ]
        schema = CollectionSchema(fields, description="Mall product and help knowledge")
        Collection(name=name, schema=schema)
    col = Collection(name)
    if not col.indexes:
        col.create_index(
            field_name="embedding",
            index_params={
                "index_type": "IVF_FLAT",
                "metric_type": "IP",
                "params": {"nlist": 128},
            },
        )
    col.load()
    return col


def _fetch_product_docs() -> list[dict[str, str]]:
    """从 mall 库拉取已上架商品文档。
    Returns:
        list[dict[str, str]]: 含 doc_id、doc_type、title、content 的文档列表。
    """
    sql = """
        SELECT p.id, p.name, p.sub_title, p.description, p.detail_desc, p.detail_html,
               p.brand_name, p.product_category_name, p.keywords
        FROM pms_product p
        WHERE p.delete_status = 0 AND p.publish_status = 1
    """
    with MallSessionLocal() as session:
        rows = session.execute(text(sql)).mappings().all()
    docs: list[dict[str, str]] = []
    for r in rows:
        content = " ".join(
            filter(
                None,
                [
                    r.get("sub_title") or "",
                    r.get("description") or "",
                    r.get("detail_desc") or "",
                    _strip_html(r.get("detail_html")),
                    f"品牌:{r.get('brand_name') or ''}",
                    f"分类:{r.get('product_category_name') or ''}",
                    f"关键词:{r.get('keywords') or ''}",
                ],
            )
        )
        content = re.sub(r"\s+", " ", content).strip()[:6000]
        if not content:
            continue
        docs.append(
            {
                "doc_id": f"product_{r['id']}",
                "doc_type": "product",
                "title": str(r["name"]),
                "content": content,
            }
        )
    return docs


def _fetch_help_docs() -> list[dict[str, str]]:
    """从 mall 库拉取 cms_help 帮助文档。
    Returns:
        list[dict[str, str]]: 含 doc_id、doc_type、title、content 的文档列表。
    """
    sql = """
        SELECT h.id, h.title, h.content, c.name AS category_name
        FROM cms_help h
        LEFT JOIN cms_help_category c ON h.category_id = c.id
        WHERE h.show_status = 1 AND h.content IS NOT NULL AND h.content != ''
    """
    with MallSessionLocal() as session:
        rows = session.execute(text(sql)).mappings().all()
    docs: list[dict[str, str]] = []
    for r in rows:
        content = _strip_html(r.get("content"))[:6000]
        if not content:
            continue
        docs.append(
            {
                "doc_id": f"help_{r['id']}",
                "doc_type": "help",
                "title": str(r.get("title") or "帮助"),
                "content": f"{r.get('category_name') or ''} {content}".strip(),
            }
        )
    return docs


def sync_knowledge() -> dict[str, int]:
    """全量同步 mall 商品与帮助文档到 Milvus。
    Returns:
        dict[str, int]: 含 indexed 字段，表示写入向量库的文档条数。
    """
    docs = _fetch_product_docs() + _fetch_help_docs()
    if not docs:
        return {"indexed": 0}
    embedder = get_embedder()
    texts = [f"{d['title']}\n{d['content']}" for d in docs]
    vectors = embedder.encode(texts, normalize_embeddings=True).tolist()
    col = ensure_collection()
    col.delete(expr='doc_id != ""')
    col.insert(
        [
            [d["doc_id"] for d in docs],
            [d["doc_type"] for d in docs],
            [d["title"][:500] for d in docs],
            [d["content"][:7900] for d in docs],
            vectors,
        ]
    )
    col.flush()
    return {"indexed": len(docs)}


def search_knowledge(query: str, top_k: int = 4) -> list[dict[str, Any]]:
    """向量检索与 query 最相关的知识片段。
    Args:
        query: 用户问题或检索关键词。
        top_k: 返回最相似的条数。
    Returns:
        list[dict[str, Any]]: 命中结果，含 doc_id、title、content、score 等。
    """
    embedder = get_embedder()
    vec = embedder.encode([query], normalize_embeddings=True).tolist()
    col = ensure_collection()
    results = col.search(
        data=vec,
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"nprobe": 16}},
        limit=top_k,
        output_fields=["doc_id", "doc_type", "title", "content"],
    )
    hits: list[dict[str, Any]] = []
    for hit in results[0]:
        hits.append(
            {
                "doc_id": hit.entity.get("doc_id"),
                "doc_type": hit.entity.get("doc_type"),
                "title": hit.entity.get("title"),
                "content": hit.entity.get("content"),
                "score": hit.score,
            }
        )
    return hits


def format_rag_context(query: str) -> str:
    """检索知识并格式化为 Agent Prompt 可用的上下文文本。
    Args:
        query: 用户问题。
    Returns:
        str: 编号的参考知识段落；无命中或 Milvus 不可用时返回提示语。
    """
    try:
        hits = search_knowledge(query)
    except Exception as exc:
        logger.warning("[rag] search failed, fallback without knowledge: %s", exc)
        return "RAG 知识库暂不可用，请依赖 Tool 查询。"
    if not hits:
        return "未检索到相关知识。"
    parts = []
    for i, h in enumerate(hits, 1):
        parts.append(f"[{i}] ({h['doc_type']}) {h['title']}\n{h['content'][:800]}")
    return "\n\n".join(parts)
