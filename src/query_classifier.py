"""
Query分类器模块

使用硅基流动的小模型API对用户Query进行分类，支持Adaptive RAG策略
"""

import os
import json
from typing import Dict, Any, Optional
from openai import OpenAI


# 硅基流动API配置（OpenAI兼容接口）
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "sk-ltwirlzumdiaprfizsxjayoxrvydmdlfrprxjyvzjykupzak")
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
CLASSIFIER_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # 可以换成更小的模型如 Qwen2-1.5B


class QueryClassifier:
    """Query分类器"""

    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        """
        初始化分类器

        Args:
            api_key: 硅基流动API Key（如果不提供则从环境变量读取）
            base_url: API Base URL
            model: 使用的模型名称
        """
        self.api_key = api_key or SILICONFLOW_API_KEY
        self.base_url = base_url or SILICONFLOW_BASE_URL
        self.model = model or CLASSIFIER_MODEL

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def classify(self, query: str) -> Dict[str, Any]:
        """
        对Query进行三分类

        Args:
            query: 用户问题

        Returns:
            分类结果字典：
            {
                "category": "no_retrieval" | "simple_lookup" | "complex_reasoning",
                "confidence": 0.9,
                "reasoning": "分类理由"
            }
        """
        prompt = self._build_classification_prompt(query)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的Query分析助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # 低温度保证稳定输出
                max_tokens=200,
            )

            result_text = response.choices[0].message.content.strip()

            # 解析JSON结果
            result = self._parse_result(result_text)
            return result

        except Exception as e:
            print(f"⚠️  Query分类失败: {e}")
            # 降级策略：返回默认分类
            return {
                "category": "simple_lookup",
                "confidence": 0.5,
                "reasoning": "分类失败，使用默认策略"
            }

    def _build_classification_prompt(self, query: str) -> str:
        """构建分类prompt"""

        prompt = f"""请分析以下用户问题，判断它属于哪种类型。

用户问题: {query}

分类规则：

1. **no_retrieval**: 无需检索知识库，直接回答
   - 特征：问候语、闲聊、简单对话
   - 示例："你好"、"谢谢"、"你是谁"

2. **simple_lookup**: 简单查找，单一事实
   - 特征：问路径规则、是否支持某功能、单个定义
   - 示例："什么是LBS？"、"路径规划支持几种模式？"

3. **complex_reasoning**: 复杂推理，需要多个信息
   - 特征：同时涉及多个模块、需要对比、需要综合多个条件
   - 示例："导航模式和主图模式的区别？"、"性能指标有哪些？"

请只返回JSON格式，不要有其他解释：
{{
    "category": "no_retrieval | simple_lookup | complex_reasoning",
    "confidence": 0.9,
    "reasoning": "简短的分类理由"
}}"""

        return prompt

    def _parse_result(self, result_text: str) -> Dict[str, Any]:
        """解析模型输出的JSON结果"""

        try:
            # 尝试直接解析JSON
            result = json.loads(result_text)

            # 验证必需字段
            if "category" not in result:
                raise ValueError("缺少category字段")

            # 验证category值
            valid_categories = ["no_retrieval", "simple_lookup", "complex_reasoning"]
            if result["category"] not in valid_categories:
                raise ValueError(f"无效的category: {result['category']}")

            # 填充默认值
            result.setdefault("confidence", 0.8)
            result.setdefault("reasoning", "")

            return result

        except json.JSONDecodeError:
            # 如果JSON解析失败，尝试提取关键信息
            result_lower = result_text.lower()

            if "no_retrieval" in result_lower:
                category = "no_retrieval"
            elif "complex_reasoning" in result_lower:
                category = "complex_reasoning"
            else:
                category = "simple_lookup"

            return {
                "category": category,
                "confidence": 0.6,
                "reasoning": "从文本中提取"
            }


def get_retrieval_strategy(classification: Dict[str, Any]) -> Dict[str, Any]:
    """
    根据Query分类结果，返回检索策略

    Args:
        classification: classify()返回的分类结果

    Returns:
        检索策略字典
    """
    category = classification["category"]

    # 基础策略模板
    strategies = {
        "no_retrieval": {
            "skip_retrieval": True,  # 跳过检索
            "retrieval_mode": None,
            "top_k": 0,
            "candidate_k": 0,
            "use_reranker": False,
            "expand_context": False,
        },

        "simple_lookup": {
            "skip_retrieval": False,
            "retrieval_mode": "dense",
            "top_k": 4,
            "candidate_k": 12,  # 候选数 = top_k * 3
            "use_reranker": False,  # 简单查询不用reranker
            "expand_context": False,
        },

        "complex_reasoning": {
            "skip_retrieval": False,
            "retrieval_mode": "dense",  # 或 "hybrid"，根据你的需要
            "top_k": 10,
            "candidate_k": 20,
            "use_reranker": True,
            "expand_context": True,  # 启用父节点/层级扩展
        },
    }

    strategy = strategies.get(category, strategies["simple_lookup"])

    # 添加分类信息
    strategy["classification"] = classification

    return strategy


# 全局单例
_classifier_instance: Optional[QueryClassifier] = None


def get_query_classifier() -> QueryClassifier:
    """获取Query分类器单例"""
    global _classifier_instance

    if _classifier_instance is None:
        _classifier_instance = QueryClassifier()

    return _classifier_instance


# 便捷函数
def classify_and_get_strategy(query: str) -> Dict[str, Any]:
    """
    一步到位：分类 + 获取策略

    Args:
        query: 用户问题

    Returns:
        包含分类和策略的完整字典
    """
    classifier = get_query_classifier()
    classification = classifier.classify(query)
    strategy = get_retrieval_strategy(classification)

    return strategy
