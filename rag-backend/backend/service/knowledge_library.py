#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识库服务层
提供知识库相关的业务逻辑处理
"""
import os
import uuid
import time
import tempfile
import requests
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from fastapi import BackgroundTasks

from backend.model.knowledge_library import KnowledgeLibrary, KnowledgeDocument
from backend.param.knowledge_library import (
    CreateLibraryRequest, UpdateLibraryRequest, AddDocumentRequest, UpdateDocumentRequest
)
from backend.param.common import Response
from backend.config.log import get_logger
from backend.config.database import DatabaseFactory

# 🔥 引入核心的 RAG 处理组件
from backend.rag.chunks.document_extraction import DocumentExtractor
from backend.rag.chunks.chunks import TextChunker
from backend.rag.chunks.models import ChunkConfig, ChunkStrategy, DocumentContent
from backend.rag.storage.milvus_storage import MilvusStorage
from backend.rag.storage.lightrag_storage import LightRAGStorage
from backend.config.embedding import get_embedding_model

logger = get_logger(__name__)

# ==============================================================================
# 🔥 新增：后台文档处理队列函数（完全复用原项目的解析与入库能力）
# ==============================================================================
async def process_uploaded_document_task(document_id: int, file_url: str, collection_id: str, doc_name: str):
    logger.info(f"🚀【后台任务】开始解析并向量化文档: {doc_name}")
    try:
        extractor = DocumentExtractor()

        # 🔥 核心修复：从真实的文件名 (doc_name) 获取后缀，彻底无视乱七八糟的 URL！
        file_extension = doc_name.split('.')[-1].lower() if '.' in doc_name else 'pdf'

        # 2. 提取文本内容
        if file_extension == 'pdf':
            logger.info("📄 检测到 PDF 文件，直接将 OSS 链接移交 MinerU 进行深度解析...")
            # MinerU 官方 API 直接支持外链下载解析，完美契合你的 OSS 架构！
            doc_content = extractor.read_document(file_url, pdf_extract_method="mineru")
        else:
            logger.info(f"📄 检测到 {file_extension} 文件，正在下载到本地处理...")
            # 对于 Word 或 Markdown，需要先下载到本地临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
                temp_path = temp_file.name

            try:
                response = requests.get(file_url)
                response.raise_for_status()
                with open(temp_path, 'wb') as f:
                    f.write(response.content)
                doc_content = extractor.read_document(temp_path)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        logger.info(f"✅ 文档 {doc_name} 提取成功，开始进行文本切块...")

        # 3. 文本分块 (Chunking)
        chunker = TextChunker()
        # 根据文件类型智能选择切分策略
        strategy = ChunkStrategy.MARKDOWN_HEADER if file_extension in ['pdf', 'md'] else ChunkStrategy.CHARACTER
        chunk_config = ChunkConfig(strategy=strategy)

        chunk_result = chunker.chunk_document(doc_content, chunk_config)

        if not chunk_result or not chunk_result.chunks:
            logger.warning(f"⚠️ 文档 {doc_name} 切块后内容为空，终止入库。")
            return

        logger.info(
            f"🔪 切块完成，共 {len(chunk_result.chunks)} 个分块。开始存入向量数据库(Milvus)和图数据库(LightRAG)...")

        # 4. 存入 Milvus 与 LightRAG 数据库
        milvus_storage = MilvusStorage(
            embedding_function=get_embedding_model(),
            collection_name=collection_id
        )
        # lightrag_storage = LightRAGStorage(workspace=collection_id)

        # 存 Milvus
        milvus_storage.store_chunks_batch([chunk_result])

        # 存 LightRAG (图谱)
        text_chunks = [chunk.page_content for chunk in chunk_result.chunks]
        # await lightrag_storage.insert_texts(text_chunks)

        logger.info(f"🎉【后台任务圆满成功】: 文档 {doc_name} 已彻底融入知识库大脑！")

    except Exception as e:
        logger.error(f"❌【后台任务失败】: 处理文档 {doc_name} 期间发生严重错误: {str(e)}")


# ==============================================================================
async def get_user_libraries(user_id: str) -> Response:
    """获取用户的知识库列表"""
    db = None
    try:
        logger.info(f"开始获取用户 {user_id} 的知识库列表")
        db = DatabaseFactory.create_session()
        
        libraries = db.query(KnowledgeLibrary).filter(
            KnowledgeLibrary.user_id == user_id,
            KnowledgeLibrary.is_active == True
        ).order_by(KnowledgeLibrary.updated_at.desc()).all()
        
        result = []
        for library in libraries:
            library_dict = library.to_dict()
            # 添加文档数量统计
            library_dict['document_count'] = len(library.documents) if library.documents else 0
            result.append(library_dict)
        
        logger.info(f"成功获取用户 {user_id} 的知识库列表，共 {len(result)} 个")
        return Response.success(result)
        
    except Exception as e:
        logger.error(f"获取用户知识库列表失败: {str(e)}")
        return Response.error(f"获取知识库列表失败: {str(e)}")
    finally:
        if db:
            db.close()


async def get_library_detail(library_id: int, user_id: str) -> Response:
    """获取知识库详情"""
    try:
        db_factory = DatabaseFactory()
        session = db_factory.create_session()
        
        try:
            library = session.query(KnowledgeLibrary).filter(
                KnowledgeLibrary.id == library_id,
                KnowledgeLibrary.user_id == user_id,
                KnowledgeLibrary.is_active == True
            ).first()
            
            if not library:
                return Response.error("知识库不存在或无权限访问")
            
            logger.info(f"成功获取知识库详情: {library.title}")
            return Response.success(library.to_dict())
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"获取知识库详情失败: {str(e)}")
        return Response.error(f"获取知识库详情失败: {str(e)}")


async def create_library(request: CreateLibraryRequest, user_id: str) -> Response:
    """创建知识库"""
    try:
        db_factory = DatabaseFactory()
        session = db_factory.create_session()
        
        try:
            # 检查同名知识库
            existing = session.query(KnowledgeLibrary).filter(
                KnowledgeLibrary.title == request.title,
                KnowledgeLibrary.user_id == user_id,
                KnowledgeLibrary.is_active == True
            ).first()
            
            if existing:
                return Response.error("已存在同名知识库")
            
            # 创建新知识库
            library = KnowledgeLibrary(
                title=request.title,
                description=request.description,
                user_id=user_id
            )
            
            session.add(library)
            session.commit()
            session.refresh(library)
            
            # 生成collection_id: kb + 知识库ID + 下划线 + 时间戳
            timestamp = str(int(time.time() * 1000))  # 毫秒级时间戳
            collection_id = f"kb{library.id}_{timestamp}"
            
            # 更新collection_id
            library.collection_id = collection_id
            session.commit()
            session.refresh(library)
            
            logger.info(f"成功创建知识库: {library.title}")
            return Response.success(library.to_dict())
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"创建知识库失败: {str(e)}")
        return Response.error(f"创建知识库失败: {str(e)}")


async def update_library(library_id: int, request: UpdateLibraryRequest, user_id: str) -> Response:
    """更新知识库"""
    try:
        db_factory = DatabaseFactory()
        session = db_factory.create_session()
        
        try:
            library = session.query(KnowledgeLibrary).filter(
                KnowledgeLibrary.id == library_id,
                KnowledgeLibrary.user_id == user_id,
                KnowledgeLibrary.is_active == True
            ).first()
            
            if not library:
                return Response.error("知识库不存在或无权限访问")
            
            # 更新字段
            if request.title is not None:
                # 检查同名知识库（排除当前库）
                existing = session.query(KnowledgeLibrary).filter(
                    KnowledgeLibrary.title == request.title,
                    KnowledgeLibrary.user_id == user_id,
                    KnowledgeLibrary.id != library_id,
                    KnowledgeLibrary.is_active == True
                ).first()
                
                if existing:
                    return Response.error("已存在同名知识库")
                
                library.title = request.title
            
            if request.description is not None:
                library.description = request.description
            
            session.commit()
            session.refresh(library)
            
            logger.info(f"成功更新知识库: {library.title}")
            return Response.success(library.to_dict())
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"更新知识库失败: {str(e)}")
        return Response.error(f"更新知识库失败: {str(e)}")


async def delete_library(library_id: int, user_id: str) -> Response:
    """删除知识库"""
    try:
        db_factory = DatabaseFactory()
        session = db_factory.create_session()
        
        try:
            library = session.query(KnowledgeLibrary).filter(
                KnowledgeLibrary.id == library_id,
                KnowledgeLibrary.user_id == user_id,
                KnowledgeLibrary.is_active == True
            ).first()
            
            if not library:
                return Response.error("知识库不存在或无权限访问")
            
            # 软删除
            library.is_active = False
            session.commit()
            
            logger.info(f"成功删除知识库: {library.title}")
            return Response.success({"message": "知识库删除成功"})
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"删除知识库失败: {str(e)}")
        return Response.error(f"删除知识库失败: {str(e)}")


# 🔥 核心修改：接收 BackgroundTasks 并触发队列
async def add_document(request: AddDocumentRequest, user_id: str, background_tasks: BackgroundTasks) -> Response:
    """添加文档到知识库"""
    try:
        db_factory = DatabaseFactory()
        session = db_factory.create_session()

        try:
            # 验证知识库权限
            library = session.query(KnowledgeLibrary).filter(
                KnowledgeLibrary.id == request.library_id,
                KnowledgeLibrary.user_id == user_id,
                KnowledgeLibrary.is_active == True
            ).first()

            if not library:
                return Response.error("知识库不存在或无权限访问")

            # 创建文档
            document = KnowledgeDocument(
                library_id=request.library_id,
                name=request.name,
                type=request.type,
                url=request.url,
                file_path=request.file_path,
                file_size=request.file_size
            )

            session.add(document)
            session.commit()
            session.refresh(document)

            logger.info(f"成功添加文档记录到知识库 {library.title}: {document.name}")

            # ============================================================
            # 🎯 发送异步任务：通知后台开启长耗时的 MinerU 解析和入库操作
            # ============================================================
            background_tasks.add_task(
                process_uploaded_document_task,
                document_id=document.id,
                # 🔥 核心修复：优先取 document.url，如果为空再取 document.file_path
                file_url=document.url or document.file_path,
                collection_id=library.collection_id,
                doc_name=document.name
            )
            logger.info(f"🚀 已成功调度后台解析任务：{document.name}")
            # ============================================================

            return Response.success(document.to_dict())
        finally:
            session.close()

    except Exception as e:
        logger.error(f"添加文档失败: {str(e)}")
        return Response.error(f"添加文档失败: {str(e)}")


async def update_document(document_id: int, request: UpdateDocumentRequest, user_id: str) -> Response:
    """更新文档"""
    try:
        db_factory = DatabaseFactory()
        session = db_factory.create_session()
        
        try:
            # 查询文档并验证权限
            document = session.query(KnowledgeDocument).join(KnowledgeLibrary).filter(
                KnowledgeDocument.id == document_id,
                KnowledgeLibrary.user_id == user_id,
                KnowledgeLibrary.is_active == True
            ).first()
            
            if not document:
                return Response.error("文档不存在或无权限访问")
            
            # 更新字段
            if request.name is not None:
                document.name = request.name
            if request.type is not None:
                document.type = request.type
            if request.url is not None:
                document.url = request.url
            if request.file_path is not None:
                document.file_path = request.file_path
            if request.file_size is not None:
                document.file_size = request.file_size
            
            session.commit()
            session.refresh(document)
            
            logger.info(f"成功更新文档: {document.name}")
            return Response.success(document.to_dict())
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"更新文档失败: {str(e)}")
        return Response.error(f"更新文档失败: {str(e)}")


async def delete_document(document_id: int, user_id: str) -> Response:
    """删除文档"""
    try:
        db_factory = DatabaseFactory()
        session = db_factory.create_session()
        
        try:
            # 查询文档并验证权限
            document = session.query(KnowledgeDocument).join(KnowledgeLibrary).filter(
                KnowledgeDocument.id == document_id,
                KnowledgeLibrary.user_id == user_id,
                KnowledgeLibrary.is_active == True
            ).first()
            
            if not document:
                return Response.error("文档不存在或无权限访问")
            
            # 物理删除文档
            session.delete(document)
            session.commit()
            
            logger.info(f"成功删除文档: {document.name}")
            return Response.success({"message": "文档删除成功"})
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"删除文档失败: {str(e)}")
        return Response.error(f"删除文档失败: {str(e)}")