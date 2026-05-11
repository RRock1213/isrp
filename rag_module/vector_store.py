"""
向量存储封装

使用 zvec 作为本地向量数据库
"""

import os
import json
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class VectorDocument:
    """向量文档"""
    doc_id: str
    text: str
    embedding: List[float]
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SearchResult:
    """检索结果"""
    doc_id: str
    text: str
    score: float
    metadata: Dict[str, Any] = None


class VectorStore:
    """
    向量存储（zvec 封装）

    使用 zvec 作为进程内向量数据库，无需独立服务
    """

    def __init__(self, db_path: str = "rag_cache/vectors", dimension: int = 1024):
        self.db_path = db_path
        self.dimension = dimension
        self._collection = None
        self._use_fallback = False
        self._initialized = False  # 标记是否已初始化
        self._current_collection_name = None  # 当前 collection 名称

        # 确保基础目录存在
        os.makedirs(db_path, exist_ok=True)

    def _init_collection(self, collection_name: str):
        """初始化 zvec collection"""
        # 如果已经初始化过同一个 collection，跳过
        if self._initialized and self._current_collection_name == collection_name:
            return

        try:
            import zvec

            schema = zvec.CollectionSchema(
                name=collection_name,
                vectors=zvec.VectorSchema("embedding", zvec.DataType.VECTOR_FP32, self.dimension),
            )

            collection_path = os.path.join(self.db_path, collection_name)

            # 如果目录已存在，尝试打开现有 collection 而不是删除
            if os.path.exists(collection_path):
                try:
                    # 尝试打开现有的 collection
                    self._collection = zvec.open(path=collection_path)
                    self._initialized = True
                    self._current_collection_name = collection_name
                    print(f"[VectorStore] Loaded existing collection: {collection_name}")
                    return
                except Exception as e:
                    # 如果打开失败，才删除并重新创建
                    print(f"[VectorStore] Cannot open existing collection: {e}, recreating...")
                    import shutil
                    try:
                        shutil.rmtree(collection_path)
                    except Exception as e2:
                        print(f"[VectorStore] Cannot remove existing collection: {e2}, using fallback")
                        self._use_fallback = True
                        self._fallback_store = {}
                        self._fallback_embeddings = {}
                        self._initialized = True
                        return

            self._collection = zvec.create_and_open(path=collection_path, schema=schema)
            self._initialized = True
            self._current_collection_name = collection_name

        except ImportError:
            print("[VectorStore] zvec not installed, using fallback memory store")
            self._use_fallback = True
            self._fallback_store = {}
            self._fallback_embeddings = {}
            self._initialized = True

        except Exception as e:
            print(f"[VectorStore] zvec init error: {e}, using fallback")
            self._use_fallback = True
            self._fallback_store = {}
            self._fallback_embeddings = {}
            self._initialized = True

    def insert(self, collection_name: str, documents: List[VectorDocument]) -> bool:
        """插入文档"""
        self._init_collection(collection_name)

        if self._use_fallback:
            return self._fallback_insert(collection_name, documents)

        try:
            import zvec

            docs = []
            for doc in documents:
                docs.append(zvec.Doc(
                    id=doc.doc_id,
                    vectors={"embedding": doc.embedding},
                    # zvec 不直接存储文本，需要在 metadata 中存储
                ))

            self._collection.insert(docs)
            return True

        except Exception as e:
            print(f"[VectorStore] Insert error: {e}")
            return False

    def _fallback_insert(self, collection_name: str, documents: List[VectorDocument]) -> bool:
        """Fallback: 内存存储"""
        if collection_name not in self._fallback_store:
            self._fallback_store[collection_name] = {}
            self._fallback_embeddings[collection_name] = {}

        for doc in documents:
            self._fallback_store[collection_name][doc.doc_id] = {
                "text": doc.text,
                "metadata": doc.metadata or {}
            }
            self._fallback_embeddings[collection_name][doc.doc_id] = doc.embedding

        return True

    def search(
        self,
        collection_name: str,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[SearchResult]:
        """向量检索"""
        self._init_collection(collection_name)

        if self._use_fallback:
            return self._fallback_search(collection_name, query_embedding, top_k)

        try:
            import zvec

            results = self._collection.query(
                zvec.VectorQuery("embedding", vector=query_embedding),
                topk=top_k
            )

            search_results = []
            for hit in results:
                # zvec 返回的是 doc_id 和 score
                search_results.append(SearchResult(
                    doc_id=hit.id,
                    text="",  # zvec 不存储原文，需要从缓存获取
                    score=hit.score,
                    metadata={}
                ))

            return search_results

        except Exception as e:
            print(f"[VectorStore] Search error: {e}")
            return []

    def _fallback_search(
        self,
        collection_name: str,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[SearchResult]:
        """Fallback: 内存检索（余弦相似度）"""
        import numpy as np

        if collection_name not in self._fallback_embeddings:
            return []

        query_vec = np.array(query_embedding)
        query_norm = np.linalg.norm(query_vec)

        if query_norm == 0:
            return []

        scores = []
        for doc_id, emb in self._fallback_embeddings[collection_name].items():
            doc_vec = np.array(emb)
            doc_norm = np.linalg.norm(doc_vec)

            if doc_norm == 0:
                continue

            similarity = np.dot(query_vec, doc_vec) / (query_norm * doc_norm)
            scores.append((doc_id, similarity))

        # 排序取 top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        top_results = scores[:top_k]

        results = []
        for doc_id, score in top_results:
            doc_info = self._fallback_store[collection_name].get(doc_id, {})
            results.append(SearchResult(
                doc_id=doc_id,
                text=doc_info.get("text", ""),
                score=float(score),
                metadata=doc_info.get("metadata", {})
            ))

        return results

    def delete_collection(self, collection_name: str) -> bool:
        """删除 collection"""
        if self._use_fallback:
            if collection_name in self._fallback_store:
                del self._fallback_store[collection_name]
                del self._fallback_embeddings[collection_name]
            return True

        # 重置 collection 引用
        self._collection = None
        self._initialized = False
        self._current_collection_name = None

        # zvec 不支持直接删除，需要删除文件
        collection_path = os.path.join(self.db_path, collection_name)
        if os.path.exists(collection_path):
            import shutil
            try:
                shutil.rmtree(collection_path)
            except Exception as e:
                # Windows 文件锁定问题，记录但不阻塞
                print(f"[VectorStore] Warning: Could not delete collection files: {e}")

        return True

    def save_index(self, collection_name: str) -> bool:
        """保存索引（zvec 自动持久化）"""
        # zvec 自动持久化到磁盘
        if self._use_fallback:
            # Fallback: 保存到 JSON
            if collection_name in self._fallback_store:
                data = {
                    "store": self._fallback_store[collection_name],
                    "embeddings": {k: list(v) for k, v in self._fallback_embeddings[collection_name].items()}
                }
                path = os.path.join(self.db_path, f"{collection_name}_fallback.json")
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

        return True

    def load_index(self, collection_name: str) -> bool:
        """加载索引"""
        if self._use_fallback:
            path = os.path.join(self.db_path, f"{collection_name}_fallback.json")
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self._fallback_store[collection_name] = data.get("store", {})
                self._fallback_embeddings[collection_name] = {
                    k: v for k, v in data.get("embeddings", {}).items()
                }
                return True

        return True

    def close(self):
        """关闭向量存储，释放资源"""
        if self._collection is not None:
            try:
                # zvec 的 collection 没有明确的 close 方法，但我们可以通过重置引用来释放
                self._collection = None
                self._initialized = False
                self._current_collection_name = None
                print("[VectorStore] 资源已释放")
            except Exception as e:
                print(f"[VectorStore] 关闭时发生错误: {e}")

    def __del__(self):
        """析构函数，确保资源释放"""
        self.close()