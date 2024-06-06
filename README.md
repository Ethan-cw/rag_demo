# rag_demo
* build_vec_index.py 离线的构建索引模块 底层用的faiss
* dialogue_manager.py 对话管理，核心对话流程管理模块
* llm_model.py 大模型调用模块
* vec_model 嵌入向量调用模块
* searcher.py 核心检索器 在线的检索，简单的向量召回 
  * 一个searcher可以有很多不同的索引和索引类型，vec_searcher是向量检索
  * 可以扩展
* main_service_online.py 在线服务是用tornado写的
* client.py 模拟客户端发起调用

## 未来可优化的点
在索引、检索和生成上都有更多精细的优化，主要的优化点会集中在：
* 索引
* 向量模型优化
* 检索后处理等模块进行优化
* 模块化，如搜索模块、记忆模块、额外生成模块、任务适配模块、对齐模块、验证模块等
  * 因为RAG本身是一个高度组织性的项目，因此在迭代过程中，是允许且需要对这些模块进行优化和调整的，可以增减、调整各个模块。
