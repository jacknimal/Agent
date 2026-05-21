import os
from datetime import timedelta
import alibabacloud_oss_v2 as oss
from threading import Lock
from dotenv import load_dotenv


class OssClientFactory:
    _instance = None
    _lock = Lock()

    @classmethod
    def get_client(cls):
        """
        获取 OSS Client 的单例实例
        """
        if cls._instance is None:
            with cls._lock:  # 线程安全
                if cls._instance is None:  # Double check
                    load_dotenv()
                    # 加载凭证
                    credentials_provider = oss.credentials.EnvironmentVariableCredentialsProvider()

                    # 默认配置
                    cfg = oss.config.load_default()
                    cfg.credentials_provider = credentials_provider

                    # V4签名特别注意：region 必须是不带 oss- 的纯地域名，例如 cn-shanghai
                    cfg.region = os.getenv('OSS_REGION', 'cn-shanghai')

                    endpoint = os.getenv('OSS_ENDPOINT')
                    if endpoint:
                        cfg.endpoint = endpoint

                    cls._instance = oss.Client(cfg)

        return cls._instance


def get_presigned_url_for_upload(bucket: str, key: str, expire_seconds: int = 3600):
    """生成预签名上传 URL"""
    client = OssClientFactory.get_client()
    clean_key = key.lstrip('/')

    # 声明只上传对象，不绑定 Content-Type 签名
    request = oss.PutObjectRequest(bucket=bucket, key=clean_key)

    pre_result = client.presign(
        request,
        method='PUT',
        expires=timedelta(seconds=expire_seconds)
    )

    final_url = pre_result.url
    # 强制修正 SDK 可能产生的双重 Bucket 路径错误
    if f"/{bucket}/" in final_url:
        final_url = final_url.replace(f"/{bucket}/", "/", 1)

    return {
        "method": "PUT",
        "url": final_url
    }


def get_presigned_url_for_download(bucket: str, key: str, expire_seconds: int = 3600):
    """生成预签名下载 URL"""
    client = OssClientFactory.get_client()
    clean_key = key.lstrip('/')

    request = oss.GetObjectRequest(bucket=bucket, key=clean_key)

    pre_result = client.presign(
        request,
        method='GET',
        expires=timedelta(seconds=expire_seconds)
    )

    final_url = pre_result.url
    # 同样对下载链接做双重 Bucket 路径修复
    if f"/{bucket}/" in final_url:
        final_url = final_url.replace(f"/{bucket}/", "/", 1)

    return {
        "method": "GET",
        "url": final_url
    }