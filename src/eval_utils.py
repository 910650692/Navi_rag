"""
RAG评估结果存储和管理
用于记录每次RAGAS评估的配置和指标，方便后续对比分析
"""

import json
import time
from pathlib import Path
from uuid import uuid4
from datetime import datetime
from typing import Dict, Any


class EvalRecorder:
    """评估结果记录器"""

    def __init__(self, log_dir: str = None):
        if log_dir is None:
            # 使用项目根目录下的data/logs
            base_dir = Path(__file__).parent.parent
            log_dir = base_dir / "data" / "logs"

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.eval_file = self.log_dir / "rag_evals.jsonl"

    def save_eval_result(self, config: Dict[str, Any], metrics: Dict[str, float], notes: str = ""):
        """
        保存一次评估结果

        Args:
            config: 评估配置，如 {"retriever": "dense", "use_reranker": True, "top_k": 6}
            metrics: RAGAS指标，如 {"context_precision": 0.85, "context_recall": 0.90, ...}
            notes: 备注说明
        """
        record = {
            "eval_id": str(uuid4())[:8],  # 短ID
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "metrics": metrics,
            "notes": notes,
        }

        # 追加写入JSONL
        with open(self.eval_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

        print(f"✅ 评估结果已保存: {self.eval_file}")
        print(f"   Eval ID: {record['eval_id']}")
        print(f"   配置: {config}")
        print(f"   指标: {metrics}")

        return record['eval_id']

    def load_eval_records(self):
        """
        加载所有评估记录

        Returns:
            List[Dict]: 评估记录列表
        """
        if not self.eval_file.exists():
            return []

        records = []
        with open(self.eval_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        return records

    def get_latest_eval(self):
        """获取最新的评估记录"""
        records = self.load_eval_records()
        if not records:
            return None
        return records[-1]


# 全局单例
_recorder: EvalRecorder = None


def get_eval_recorder() -> EvalRecorder:
    """获取评估记录器（单例）"""
    global _recorder
    if _recorder is None:
        _recorder = EvalRecorder()
    return _recorder
