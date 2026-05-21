/**
 * 知识库API服务模块
 * 提供知识库和文档的增删改查功能
 */

import { httpClient } from './config.js'

/**
 * 知识库API服务类
 */
class KnowledgeLibraryAPI {
  
  /**
   * 获取用户的知识库列表
   * @returns {Promise} API响应
   */
  async getLibraries() {
    try {
      const response = await httpClient.get('/api/knowledge/libraries')
      return response
    } catch (error) {
      console.error('获取知识库列表失败:', error)
      throw error
    }
  }

  /**
   * 获取知识库详情
   * @param {number} libraryId - 知识库ID
   * @returns {Promise} API响应
   */
  async getLibraryDetail(libraryId) {
    try {
      const response = await httpClient.get(`/api/knowledge/libraries/${libraryId}`)
      return response
    } catch (error) {
      console.error('获取知识库详情失败:', error)
      throw error
    }
  }

  /**
   * 创建知识库
   * @param {Object} libraryData - 知识库数据
   * @param {string} libraryData.title - 知识库标题
   * @param {string} libraryData.description - 知识库描述
   * @returns {Promise} API响应
   */
  async createLibrary(libraryData) {
    try {
      const response = await httpClient.post('/api/knowledge/libraries', libraryData)
      return response
    } catch (error) {
      console.error('创建知识库失败:', error)
      throw error
    }
  }

  /**
   * 更新知识库
   * @param {number} libraryId - 知识库ID
   * @param {Object} libraryData - 知识库数据
   * @returns {Promise} API响应
   */
  async updateLibrary(libraryId, libraryData) {
    try {
      const response = await httpClient.put(`/api/knowledge/libraries/${libraryId}`, libraryData)
      return response
    } catch (error) {
      console.error('更新知识库失败:', error)
      throw error
    }
  }

  /**
   * 删除知识库
   * @param {number} libraryId - 知识库ID
   * @returns {Promise} API响应
   */
  async deleteLibrary(libraryId) {
    try {
      const response = await httpClient.delete(`/api/knowledge/libraries/${libraryId}`)
      return response
    } catch (error) {
      console.error('删除知识库失败:', error)
      throw error
    }
  }

  /**
   * 添加文档到知识库
   * @param {Object} documentData - 文档数据
   * @param {number} documentData.library_id - 知识库ID
   * @param {string} documentData.title - 文档标题
   * @param {string} documentData.content - 文档内容
   * @returns {Promise} API响应
   */
  async addDocument(documentData) {
    try {
      const response = await httpClient.post('/api/knowledge/documents', documentData)
      return response
    } catch (error) {
      console.error('添加文档失败:', error)
      throw error
    }
  }

  /**
   * 更新文档
   * @param {number} documentId - 文档ID
   * @param {Object} documentData - 文档数据
   * @returns {Promise} API响应
   */
  async updateDocument(documentId, documentData) {
    try {
      const response = await httpClient.put(`/api/knowledge/documents/${documentId}`, documentData)
      return response
    } catch (error) {
      console.error('更新文档失败:', error)
      throw error
    }
  }

  /**
   * 删除文档
   * @param {number} documentId - 文档ID
   * @returns {Promise} API响应
   */
  async deleteDocument(documentId) {
    try {
      const response = await httpClient.delete(`/api/knowledge/documents/${documentId}`)
      return response
    } catch (error) {
      console.error('删除文档失败:', error)
      throw error
    }
  }

  /**
   * 获取文件上传URL
   * @param {string} documentName - 文档名称
   * @returns {Promise} API响应
   */
  /**
   * 获取文件上传URL
   * @param {string} documentName - 文档名称
   * @returns {Promise} API响应
   */
  async getUploadUrl(documentName) {
    try {
      const response = await httpClient.post('/api/knowledge/upload-url', {
        document_name: documentName
      })

      // 【防报错拦截】：如果后端返回的 data 是一个对象（包含 url），
      // 我们强制把它降维剥离成纯字符串！这是为了迎合 Vue 组件里的 .split() 调用。
      if (response.data && typeof response.data === 'object' && response.data.url) {
        response.data = response.data.url;
      }

      return response
    } catch (error) {
      console.error('获取上传URL失败:', error)
      throw error
    }
  }

  /**
   * 上传文件到OSS
   * @param {string|Object} uploadUrl - 上传URL (兼容对象结构)
   * @param {File} file - 文件对象
   * @returns {Promise} 上传响应
   */
  async uploadFileToOSS(uploadUrl, file) {
    try {
      // 1. 安全提取真实的上传链接（增强版提取逻辑，适配拦截后的字符串）
      let finalUrl = '';
      if (typeof uploadUrl === 'string') {
        finalUrl = uploadUrl;
      } else if (typeof uploadUrl === 'object') {
        // 如果上面拦截成功，uploadUrl.data 已经是字符串了
        if (typeof uploadUrl.data === 'string') {
          finalUrl = uploadUrl.data;
        } else {
          finalUrl = uploadUrl.data?.url || uploadUrl.url;
        }
      }

      // 2. 转为无头二进制流
      const buffer = await file.arrayBuffer();
      console.log('🚀 准备使用纯二进制流上传，目标地址:', finalUrl);

      const response = await fetch(finalUrl, {
        method: 'PUT',
        body: buffer,
        headers: {} // 保持空请求头
      });

      if (!response.ok) {
        const errText = await response.text();
        console.error("OSS返回的错误详情:", errText);
        throw new Error(`上传失败: ${response.status} ${response.statusText}`);
      }

      console.log('✅ 文件上传到 OSS 成功！');

      // 【终极保险】：强行篡改外部对象的 data 属性为纯字符串，防止 Vue 继续报错
      if (typeof uploadUrl === 'object') {
        uploadUrl.data = finalUrl;
      }

      return {
        success: true,
        url: finalUrl.split('?')[0],
        data: finalUrl
      };

    } catch (error) {
      console.error('❌ 上传文件到OSS失败:', error);
      throw error;
    }
  }

  /**
   * 爬取网站内容
   * @param {Object} crawlData - 爬取数据
   * @param {string} crawlData.url - 网站URL
   * @param {number} crawlData.library_id - 知识库ID
   * @param {number} [crawlData.max_pages] - 最大页面数
   * @returns {Promise} API响应
   */
  async crawlSite(crawlData) {
    try {
      const response = await httpClient.post('/api/crawl/site', crawlData)
      return response
    } catch (error) {
      console.error('爬取网站失败:', error)
      throw error
    }
  }

  /**
   * 获取知识图谱数据
   * @param {string} collectionId - 集合ID
   * @param {string} label - 标签过滤器
   * @returns {Promise} API响应
   */
  async getKnowledgeGraph(collectionId, label = '*') {
    try {
      const response = await httpClient.get(`/api/visual/graph/${collectionId}`, {
        label: label
      })
      return response
    } catch (error) {
      console.error('获取知识图谱失败:', error)
      throw error
    }
  }
}

// 创建并导出API实例
export const knowledgeAPI = new KnowledgeLibraryAPI()

// 导出默认实例
export default knowledgeAPI