"""
断点续传机制实现

支持评估任务的断点续传功能：
- 任务状态持久化
- 进度记录
- 任务恢复
- 数据一致性保证
- 失败样本管理
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime


class CheckpointManager:
    """断点续传管理器"""

    def __init__(self, output_file: str):
        """
        初始化断点管理器

        参数：
            output_file: 输出文件路径
        """
        self.output_file = output_file
        self.checkpoint_file = self._get_checkpoint_file()
        self.failed_samples_file = self._get_failed_samples_file()

    def _get_checkpoint_file(self) -> str:
        """获取断点文件路径"""
        output_path = Path(self.output_file)
        checkpoint_path = output_path.parent / f".{output_path.stem}_checkpoint.json"
        return str(checkpoint_path)
    
    def _get_failed_samples_file(self) -> str:
        """获取失败样本文件路径"""
        output_path = Path(self.output_file)
        failed_path = output_path.parent / f".{output_path.stem}_failed_samples.jsonl"
        return str(failed_path)

    def save_checkpoint(
        self,
        total_samples: int,
        processed_samples: int,
        failed_samples: int,
        last_processed_index: int,
        start_time: datetime,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        保存断点信息

        参数：
            total_samples: 总样本数
            processed_samples: 已处理样本数
            failed_samples: 失败样本数
            last_processed_index: 最后处理的样本索引
            start_time: 任务开始时间
            metadata: 额外的元数据
        """
        checkpoint_data = {
            "output_file": self.output_file,
            "total_samples": total_samples,
            "processed_samples": processed_samples,
            "failed_samples": failed_samples,
            "last_processed_index": last_processed_index,
            "start_time": start_time.isoformat(),
            "last_checkpoint_time": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        # 保存到断点文件
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
    
    def save_failed_sample(self, prompt: str, error: str, retry_count: int = 0, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        保存失败样本信息
        
        参数：
            prompt: 样本的 prompt
            error: 错误信息
            retry_count: 重试次数
            metadata: 额外的元数据
        """
        failed_sample = {
            "prompt": prompt,
            "prompt_hash": hash(prompt),
            "error": error,
            "retry_count": retry_count,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        
        # 追加到失败样本文件
        with open(self.failed_samples_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(failed_sample, ensure_ascii=False) + '\n')
    
    def load_failed_samples(self) -> List[Dict[str, Any]]:
        """
        加载失败样本列表
        
        返回：
            失败样本列表
        """
        if not os.path.exists(self.failed_samples_file):
            return []
        
        failed_samples = []
        try:
            with open(self.failed_samples_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        failed_samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"警告：加载失败样本文件失败: {e}")
        
        return failed_samples
    
    def get_failed_samples_before_index(self, current_index: int) -> List[Dict[str, Any]]:
        """
        获取当前进度之前的所有失败样本
        
        参数：
            current_index: 当前进度索引
            
        返回：
            失败样本列表
        """
        failed_samples = self.load_failed_samples()
        
        # 筛选当前进度之前的失败样本
        failed_before = [
            sample for sample in failed_samples
            if sample.get('metadata', {}).get('last_index', 0) < current_index
        ]
        
        return failed_before
    
    def clear_failed_samples(self) -> None:
        """清空失败样本文件"""
        if os.path.exists(self.failed_samples_file):
            os.remove(self.failed_samples_file)

    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        加载断点信息

        返回：
            断点信息字典，如果不存在则返回 None
        """
        if not os.path.exists(self.checkpoint_file):
            return None

        try:
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)

            # 验证断点文件是否匹配当前输出文件（比较绝对路径）
            checkpoint_output = checkpoint_data.get("output_file", "")
            current_output = self.output_file
            
            # 将两个路径都转换为绝对路径进行比较
            checkpoint_output_abs = str(Path(checkpoint_output).resolve()) if checkpoint_output else ""
            current_output_abs = str(Path(current_output).resolve()) if current_output else ""
            
            if checkpoint_output_abs != current_output_abs:
                print(f"警告：断点文件与当前输出文件不匹配，忽略断点")
                print(f"  断点记录: {checkpoint_output}")
                print(f"  当前文件: {current_output}")
                return None

            return checkpoint_data

        except Exception as e:
            print(f"警告：加载断点文件失败: {e}")
            return None

    def delete_checkpoint(self) -> None:
        """删除断点文件"""
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)

    def has_checkpoint(self) -> bool:
        """检查是否存在断点"""
        return os.path.exists(self.checkpoint_file)

    def get_processed_indices(self) -> set:
        """
        从输出文件中获取已处理的样本索引

        返回：
            已处理样本索引的集合
        """
        if not os.path.exists(self.output_file):
            return set()

        processed_indices = set()

        try:
            with open(self.output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        # 检查是否是统计摘要行
                        if "total_score" in data and "total_samples" in data:
                            continue
                        if "mean_score" in data and "total_samples" in data:
                            continue
                        # 检查是否有 prompt（有效的评估结果）
                        # 支持 scores 和 score 两种格式
                        if "prompt" in data and ("scores" in data or "score" in data):
                            # 使用 prompt 的哈希作为唯一标识
                            prompt_hash = hash(data["prompt"])
                            processed_indices.add(prompt_hash)
                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            print(f"警告：读取输出文件失败: {e}")

        return processed_indices


class TaskProgressTracker:
    """任务进度跟踪器"""

    def __init__(self, total_samples: int, output_file: str):
        """
        初始化进度跟踪器

        参数：
            total_samples: 总样本数
            output_file: 输出文件路径
        """
        self.total_samples = total_samples
        self.output_file = output_file
        self.checkpoint_manager = CheckpointManager(output_file)
        self.start_time = datetime.now()
        self.processed_count = 0
        self.failed_count = 0
        self.processed_indices = set()

    def load_progress(self) -> bool:
        """
        加载之前的进度

        返回：
            是否成功加载进度
        """
        checkpoint = self.checkpoint_manager.load_checkpoint()

        if checkpoint is None:
            return False

        # 从输出文件中获取已处理的索引
        self.processed_indices = self.checkpoint_manager.get_processed_indices()

        # 恢复计数器
        self.processed_count = checkpoint.get("processed_samples", 0)
        self.failed_count = checkpoint.get("failed_samples", 0)

        # 恢复开始时间
        try:
            self.start_time = datetime.fromisoformat(checkpoint.get("start_time", datetime.now().isoformat()))
        except:
            self.start_time = datetime.now()

        print(f"\n从断点恢复进度:")
        print(f"  - 总样本数: {self.total_samples}")
        print(f"  - 已处理: {self.processed_count}")
        print(f"  - 失败: {self.failed_count}")
        print(f"  - 剩余: {self.total_samples - self.processed_count}")

        return True

    def save_progress(self, last_processed_index: int) -> None:
        """
        保存当前进度

        参数：
            last_processed_index: 最后处理的样本索引
        """
        self.checkpoint_manager.save_checkpoint(
            total_samples=self.total_samples,
            processed_samples=self.processed_count,
            failed_samples=self.failed_count,
            last_processed_index=last_processed_index,
            start_time=self.start_time,
        )

    def is_processed(self, prompt: str) -> bool:
        """
        检查样本是否已处理

        参数：
            prompt: 样本的 prompt

        返回：
            是否已处理
        """
        prompt_hash = hash(prompt)
        return prompt_hash in self.processed_indices

    def mark_processed(self, prompt: str, success: bool = True, error: str = None, retry_count: int = 0, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        标记样本为已处理

        参数：
            prompt: 样本的 prompt
            success: 是否成功处理
            error: 错误信息（如果失败）
            retry_count: 重试次数
            metadata: 额外的元数据
        """
        prompt_hash = hash(prompt)
        self.processed_indices.add(prompt_hash)

        if success:
            self.processed_count += 1
        else:
            self.failed_count += 1
            # 保存失败样本信息
            self.checkpoint_manager.save_failed_sample(
                prompt=prompt,
                error=error or "Unknown error",
                retry_count=retry_count,
                metadata=metadata or {}
            )
    
    def get_failed_samples_to_retry(self, current_index: int) -> List[Dict[str, Any]]:
        """
        获取当前进度之前需要重试的失败样本
        
        参数：
            current_index: 当前进度索引
            
        返回：
            需要重试的失败样本列表
        """
        return self.checkpoint_manager.get_failed_samples_before_index(current_index)

    def get_progress(self) -> Dict[str, Any]:
        """
        获取当前进度

        返回：
            进度信息字典
        """
        elapsed_time = (datetime.now() - self.start_time).total_seconds()

        return {
            "total_samples": self.total_samples,
            "processed_samples": self.processed_count,
            "failed_samples": self.failed_count,
            "success_samples": self.processed_count - self.failed_count,
            "remaining_samples": self.total_samples - self.processed_count,
            "progress_percentage": (self.processed_count / self.total_samples * 100) if self.total_samples > 0 else 0,
            "elapsed_time_seconds": elapsed_time,
            "elapsed_time_formatted": self._format_time(elapsed_time),
        }

    def _format_time(self, seconds: float) -> str:
        """格式化时间"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def finish(self) -> None:
        """任务完成，清理断点文件"""
        self.checkpoint_manager.delete_checkpoint()
