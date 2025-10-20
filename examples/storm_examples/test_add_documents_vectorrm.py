import os
import argparse
import pandas as pd

from knowledge_storm.rm import VectorRM
from knowledge_storm.utils import QdrantVectorStoreManager


def generate_sample_csv(csv_path: str):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "content": "机器学习是人工智能的一个分支，专注于让计算机从数据中学习模式并进行预测或决策。",
                "title": "机器学习简介",
                "url": "doc-ml-intro",
                "description": "ML 概念与应用概览",
            },
            {
                "content": "深度学习使用多层神经网络对复杂函数进行逼近，常用于视觉、语音和自然语言处理。",
                "title": "深度学习基础",
                "url": "doc-dl-basics",
                "description": "DL 基础与常见网络结构",
            },
            {
                "content": "卷积神经网络通过局部感受野和权重共享处理图像等具有空间结构的数据，有效降低参数数量。",
                "title": "卷积神经网络",
                "url": "doc-cnn",
                "description": "CNN 关键思想与优势",
            },
        ]
    )
    df.to_csv(csv_path, index=False)


def build_offline_vector_store(
    csv_path: str,
    vector_store_dir: str,
    collection_name: str,
    embedding_model: str,
    device: str,
    batch_size: int,
    chunk_size: int,
    chunk_overlap: int,
):
    QdrantVectorStoreManager.create_or_update_vector_store(
        collection_name=collection_name,
        vector_db_mode="offline",
        file_path=csv_path,
        content_column="content",
        title_column="title",
        url_column="url",
        desc_column="description",
        batch_size=batch_size,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        vector_store_path=vector_store_dir,
        embedding_model=embedding_model,
        device=device,
    )


def verify_vectorrm_query(
    vector_store_dir: str,
    collection_name: str,
    embedding_model: str,
    device: str,
    top_k: int,
):
    rm = VectorRM(
        collection_name=collection_name,
        embedding_model=embedding_model,
        device=device,
        k=top_k,
    )
    rm.init_offline_vector_db(vector_store_path=vector_store_dir)

    print("=== Vector Store Initialized ===")
    try:
        count = rm.get_vector_count()
        print(f"Vector count in collection '{collection_name}': {count}")
    except Exception:
        pass

    test_queries = [
        "什么是机器学习",
        "深度学习的基本原理是什么",
        "卷积神经网络解决了什么问题",
    ]

    for q in test_queries:
        results = rm.forward(q)
        print(f"\nQuery: {q}")
        print(f"Top-{top_k} results: {len(results)}")
        for i, r in enumerate(results, start=1):
            title = r.get("title", "")
            url = r.get("url", "")
            snippet = (r.get("snippets", [""])[0] or "")[:120].replace("\n", " ")
            print(f"  {i}. {title} | {url}")
            print(f"     snippet: {snippet}...")


def main():
    parser = argparse.ArgumentParser(
        description="Test adding documents to VectorRM and verify retrieval."
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="./data/test_docs.csv",
        help="Path to output CSV containing sample documents.",
    )
    parser.add_argument(
        "--vector-store-dir",
        type=str,
        default="./vector_store_test",
        help="Directory for offline Qdrant vector store.",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="custom_kb_test",
        help="Qdrant collection name.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="BAAI/bge-m3",
        help="HF embedding model name.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Embedding device: cpu, cuda, mps, etc.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for adding documents to vector store.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Chunk size for splitting documents.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Chunk overlap for splitting documents.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Top-k for retrieval verification.",
    )
    args = parser.parse_args()

    print("=== Generating sample CSV ===")
    generate_sample_csv(args.csv_path)

    print("=== Building offline vector store ===")
    build_offline_vector_store(
        csv_path=args.csv_path,
        vector_store_dir=args.vector_store_dir,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        device=args.device,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    print("=== Verifying retrieval with VectorRM ===")
    verify_vectorrm_query(
        vector_store_dir=args.vector_store_dir,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        device=args.device,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()


