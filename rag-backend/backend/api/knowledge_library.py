from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from backend.param.knowledge_library import (
    CreateLibraryRequest, UpdateLibraryRequest, AddDocumentRequest, UpdateDocumentRequest
)
import os
from backend.param.common import Response
from backend.service import knowledge_library as library_service
from backend.config.log import get_logger
from backend.config.dependencies import get_current_user
from typing import Dict
from backend.config.oss import get_presigned_url_for_upload
logger = get_logger(__name__)

router = APIRouter(
    prefix="/knowledge",
    tags=["KNOWLEDGE_LIBRARY"]
)


@router.get("/libraries")
async def get_libraries(current_user: int = Depends(get_current_user)):
    """获取用户的知识库列表"""
    logger.info(f"用户 {current_user} 请求获取知识库列表")
    return await library_service.get_user_libraries(current_user)


@router.get("/libraries/{library_id}")
async def get_library(library_id: int, current_user: int = Depends(get_current_user)):
    """获取知识库详情"""
    logger.info(f"用户 {current_user} 请求获取知识库详情: {library_id}")
    return await library_service.get_library_detail(library_id, current_user)


@router.post("/libraries")
async def create_library(request: CreateLibraryRequest, current_user: int = Depends(get_current_user)):
    """创建知识库"""
    logger.info(f"用户 {current_user} 请求创建知识库: {request.title}")
    return await library_service.create_library(request, current_user)


@router.put("/libraries/{library_id}")
async def update_library(
    library_id: int,
    request: UpdateLibraryRequest,
    current_user: int = Depends(get_current_user)
):
    """更新知识库"""
    logger.info(f"用户 {current_user} 请求更新知识库: {library_id}")
    return await library_service.update_library(library_id, request, current_user)


@router.delete("/libraries/{library_id}")
async def delete_library(library_id: int, current_user: int = Depends(get_current_user)):
    """删除知识库"""
    logger.info(f"用户 {current_user} 请求删除知识库: {library_id}")
    return await library_service.delete_library(library_id, current_user)


@router.post("/documents")
async def add_document(
    request: AddDocumentRequest,
    background_tasks: BackgroundTasks,
    current_user: int = Depends(get_current_user)
):
    """添加文档到知识库"""
    logger.info(f"用户 {current_user} 请求添加文档到知识库: {request.library_id}")
    return await library_service.add_document(request, current_user, background_tasks)


@router.put("/documents/{document_id}")
async def update_document(
    document_id: int,
    request: UpdateDocumentRequest,
    current_user: int = Depends(get_current_user)
):
    """更新文档"""
    logger.info(f"用户 {current_user} 请求更新文档: {document_id}")
    return await library_service.update_document(document_id, request, current_user)


@router.delete("/documents/{document_id}")
async def delete_document(document_id: int, current_user: int = Depends(get_current_user)):
    """删除文档"""
    logger.info(f"用户 {current_user} 请求删除文档: {document_id}")
    return await library_service.delete_document(document_id, current_user)


@router.post("/upload-url")
async def get_upload_url(
        payload: Dict,
        current_user: int = Depends(get_current_user)
):
    """获取 OSS 预签名上传链接 (前端直传文件到OSS使用)"""
    logger.info(f"用户 {current_user} 请求获取OSS上传链接: {payload}")
    try:
        filename = payload.get("filename") or payload.get("file_name") or payload.get("name") or "unnamed_upload_file"

        # 完美做法：动态读取 .env 配置文件中的 OSS_BUCKET_NAME
        bucket_name = os.getenv("OSS_BUCKET_NAME")

        # 做一个安全拦截：如果管理员忘了配置环境，直接报错提示
        if not bucket_name:
            return Response.error("服务器环境变量中未配置 OSS_BUCKET_NAME")

        # 传入动态读取的 bucket_name
        result = get_presigned_url_for_upload(bucket=bucket_name, key=filename)

        return Response.success(result)
    except Exception as e:
        logger.error(f"获取OSS上传链接失败: {str(e)}")
        return Response.error(f"获取上传链接失败: {str(e)}")