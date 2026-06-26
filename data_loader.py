# ==============================================
# 文件名：universal_data_loader.py
# 基础功能：实现同时兼容【本地文件路径】+【HTTP/HTTPS网络URL】的通用数据加载器
# 核心学习知识点：
# 1. Chroma自定义DataLoader抽象类继承、多类型URI路由分发逻辑
# 2. 本地文件二进制安全读取、网络资源异常捕获、超时容错处理
# 3. 多模态知识库uris、data字段使用前提与底层加载原理
# 适用场景：图片/PDF等本地文件入库、网络图片链接向量库、图文RAG系统
# 使用方法：创建集合时传入本加载器，add阶段填入本地路径或网络链接，即可通过get读取data二进制
# 进阶说明：
# 1. 不绑定DataLoader无法读取uris、data字段；绑定后自动根据uri类型分发加载逻辑
# 2. 路径安全校验防止路径穿越、网络请求做超时+状态码异常拦截
# ==============================================
import os
import requests
from typing import List


from chromadb.utils.data_loaders import DataLoader
from pathlib import Path

class UniversalFileDataLoader(DataLoader):
    """
    【基础功能】通用资源加载器，自动识别本地文件绝对路径 / 网络HTTP/HTTPS链接
    根据传入的uris批量读取资源二进制字节流，供Chroma的data字段返回使用
    【学习知识点】
        1. 抽象基类继承：必须重写load(uris)方法，输入URI列表，返回等长bytes列表
        2. 类型路由：判断uri是否以http开头区分本地/网络资源
        3. 工程容错：文件存在校验、路径安全校验、网络超时、请求异常捕获
    参数：
        timeout: int
            【基础释义】网络请求超时时间，单位秒，默认10秒
            【进阶释义】防止网络卡死阻塞整个查询流程，生产环境建议5~10秒
        root_allowed_dir: str | None
            【基础释义】允许读取的本地根目录，默认None不限制；传入路径则做路径安全校验
            【进阶释义】防御路径穿越漏洞，仅允许读取项目指定目录内的本地文件
    返回值：
        load方法：返回和uris顺序一一对应的文件二进制bytes列表
    调用示例：
        # 示例1：默认配置，本地任意文件+网络链接均可加载
        loader = UniversalFileDataLoader()

        # 示例2：限制仅能读取项目根目录下文件，网络链接超时5秒
        loader = UniversalFileDataLoader(timeout=5, root_allowed_dir="./")
    同场景常用替代方案：
        1. 仅本地加载：精简版LocalFileDataLoader，无网络请求逻辑，内存占用更低
        2. 仅网络加载：HttpFileDataLoader，只处理URL，适合云端图片向量库
    注意事项：
        1. 集合一旦绑定该加载器，后续所有uris必须是合法本地路径或http/https链接
        2. 相对路径会自动转为绝对路径做安全校验，建议统一传入绝对路径
        3. 批量URI中有任意一个资源加载失败会直接抛出异常，建议提前校验文件可用性
    """
    def __init__(self, timeout: int = 10, root_allowed_dir: str = None):
        # 网络请求超时时间
        self.timeout = timeout
        # 允许访问的本地根目录，用于路径安全校验
        self.root_allowed_dir = Path(root_allowed_dir).resolve() if root_allowed_dir else None

    def _is_http_url(self, uri: str) -> bool:
        """私有方法：判断当前URI是否为网络链接"""
        return uri.startswith(("http://", "https://"))

    def _load_local_file(self, uri: str) -> bytes:
        """加载本地文件二进制，附带路径安全校验"""
        file_path = Path(uri).resolve()

        # 如果配置了允许的根目录，校验文件不能在根目录之外（防御路径穿越）
        if self.root_allowed_dir is not None:
            try:
                file_path.relative_to(self.root_allowed_dir)
            except ValueError:
                raise PermissionError(
                    f"路径安全拦截：禁止读取根目录 {self.root_allowed_dir} 以外的文件：{uri}"
                )

        # 校验文件必须真实存在且是文件，不能是文件夹
        if not file_path.exists():
            raise FileNotFoundError(f"本地文件不存在：{uri}")
        if not file_path.is_file():
            raise IsADirectoryError(f"传入路径不是文件：{uri}")

        # 二进制模式读取文件
        with open(file_path, "rb") as f:
            return f.read()

    def _load_http_resource(self, uri: str) -> bytes:
        """请求网络资源，捕获网络异常"""
        try:
            resp = requests.get(uri, timeout=self.timeout)
            # 主动抛出4xx、5xx类HTTP错误
            resp.raise_for_status()
            return resp.content
        except requests.exceptions.Timeout:
            raise TimeoutError(f"网络请求超时（{self.timeout}s）：{uri}")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"网络资源请求失败 {uri}，错误详情：{str(e)}")

    def load(self, uris: List[str]) -> List[bytes]:
        """
        【重写抽象方法】批量加载多个URI资源
        入参：多个本地路径/网络URL组成的列表
        返回：顺序一一对应的二进制字节列表
        """
        bytes_result = []
        for uri in uris:
            if self._is_http_url(uri):
                # 网络资源分支
                file_bytes = self._load_http_resource(uri)
            else:
                # 本地文件分支
                file_bytes = self._load_local_file(uri)
            bytes_result.append(file_bytes)
        return bytes_result