"""
ISRP 核心规划编排器

管理完整的 outline 生成管线，按三阶段串联所有处理器：

  Phase I — 预处理
    PromptCompressor         查询压缩（超长 prompt 触发）
    QueryStyleRecognition    风格识别 + 显式章节框架检测（快捷路径）

  Phase II — 大纲树生成
    RootNodeEstablishment    根节点建立（意图发散 + 串行反思迭代）
    GlobalAbstract           提取全局摘要（指导深层节点生成）
    DFSExpansionEngine       深度优先节点扩展（纵向 + 横向 + 概念覆盖池）
    ChapterLevelValidation   章节级检查点验证（完成一级章节子树后触发）
    GlobalValidation         全局验证（编号、风格、结构完整性校验）

  Phase III — 大纲树优化
    Anti-Forgetting Injection  每次 LLM 调用前强制注入全局记忆 + 局部上下文
    DescriptionOptimize        批量精修节点描述（specificity + guidance >= 8）
    WordCountAllocation        BFS 逐层权重分配字数，合并/拆分调整
    Format                     标准化段落输出（AgentWrite 兼容格式）

"""
import asyncio
import sys
import traceback
from typing import Dict, Any, Optional

from .schemas import (
    OutlineNode, PlanSession, detect_language
)
from .engines import MAX_DEPTH, MAX_WIDTH, PROMPT_COMPRESS_THRESHOLD, PROMPT_COMPRESS_TARGET
from .prompts import PromptTemplates, MAX_ITERATIONS
from .utils.llm_client import LLMClient
from .utils.node_ops import NodeOperations
from .utils.tree_ops import TreeOperations
from .utils.prompt_compressor import PromptCompressor
from .utils.hash_utils import generate_sample_id  # 统一使用哈希工具
from .steps import (
    QueryStyleRecognition,
    RootNodeEstablishment,
    GlobalAbstract,
    DFSExpansionEngine,
    ChapterLevelValidation,
    DescriptionOptimize,
    WordCountAllocation,
    GlobalValidation,
    Format
)

# 类型提示
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .evaluators.base import BaseEvaluator


# ============================================================================
# 实时输出辅助函数
# ============================================================================

def _flush_stdout():
    """强制刷新 stdout，确保实时输出"""
    sys.stdout.flush()


def _print_and_flush(msg: str):
    """打印并立即刷新输出"""
    print(msg)
    _flush_stdout()


# ============================================================================
# ISRPPlanner 类
# ============================================================================


class ISRPPlanner:
    """
    ISRP 核心规划器

    串行自我纠正与延迟约束树状规划器
    """

    def __init__(
        self,
        client,
        max_depth: int = MAX_DEPTH,
        max_iterations: int = MAX_ITERATIONS,
        enable_logging: bool = True,
        language: str = None,
        persist_file: str = None,
        evaluator: 'BaseEvaluator' = None,
        rag_service = None,  # RAG 服务（可选）
        max_width: int = None  # 生成宽度上限，None 表示使用 engines.MAX_WIDTH 默认值
    ):
        """
        初始化规划器

        Args:
            client: API 客户端
            max_depth: 最大树深度
            max_iterations: 串行迭代上限
            enable_logging: 是否启用日志
            language: 语言代码
            persist_file: LLM 调用持久化文件路径
            evaluator: 评估器实例
            rag_service: RAG 服务实例（可选，用于长 prompt 检索增强）
            max_width: DFS 扩展时每层最大子节点数
        """
        self.client = client
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.enable_logging = enable_logging
        self.language = language
        self.persist_file = persist_file
        self.evaluator = evaluator
        self.rag_service = rag_service  # RAG 服务
        self.max_width = max_width

        # 初始化工具
        self.llm = LLMClient(client, persist_file, enable_logging)
        self.prompts = PromptTemplates(language or "zh")
        self.node_ops = NodeOperations()
        self.tree_ops = TreeOperations()
        self.prompt_compressor = PromptCompressor(self.llm, self.prompts, enable_logging)

        # 会话
        self.session: Optional[PlanSession] = None

    async def generate_plan(
        self,
        user_prompt: str,
        total_words: int = 10000,
        sample_id: str = None  # 样本 ID（用于 RAG）
    ) -> str:
        """
        生成大纲（主入口 - 异步版）

        Args:
            user_prompt: 用户指令
            total_words: 目标字数
            sample_id: 样本 ID（用于 RAG 索引和检索）

        Returns:
            LongWriter 标准格式的大纲
        """
        if self.enable_logging:
            _print_and_flush("\n" + "=" * 70)
            _print_and_flush("ISRP - Serial Self-Correction Planner")
            _print_and_flush("=" * 70)
            # 显示 prompt 信息（不截断）
            if len(user_prompt) > PROMPT_COMPRESS_THRESHOLD:
                _print_and_flush(f"User prompt: {len(user_prompt)} chars (will be compressed)")
            else:
                _print_and_flush(f"User prompt: {user_prompt}")
            _print_and_flush(f"Target words: {total_words}")

        # 生成 sample_id（如果未提供）
        if not sample_id:
            sample_id = generate_sample_id(user_prompt)

        try:
            # 初始化会话
            self.session = PlanSession()
            self.session.original_prompt = user_prompt
            self.session.global_target_words = total_words
            self.session._sample_id = sample_id  # 存储 sample_id 供根节点建立使用

            # RAG 索引（在 Prompt 压缩前）
            if self.rag_service and self.rag_service.is_enabled():
                if self.rag_service.should_use_rag(user_prompt):
                    if self.enable_logging:
                        _print_and_flush(f"\n[RAG] Indexing prompt for sample: {sample_id}")
                    rag_result = self.rag_service.index_prompt(sample_id, user_prompt)
                    if self.enable_logging:
                        _print_and_flush(f"  Chunks: {rag_result.chunk_count}, Cached: {rag_result.cached}")

            # 预处理: Prompt 智能压缩
            if len(user_prompt) > PROMPT_COMPRESS_THRESHOLD:
                if self.enable_logging:
                    _print_and_flush(f"\n[预处理] Prompt Compression")
                    _print_and_flush(f"  Original length: {len(user_prompt)} chars")
                    _print_and_flush(f"  Triggering smart compression...")

                compressed = self.prompt_compressor.compress(
                    user_prompt,
                    target_length=PROMPT_COMPRESS_TARGET,
                    language=self.language
                )

                if compressed:
                    self.session.compressed_prompt = compressed
                    self.session.effective_prompt = compressed.compressed_prompt
                else:
                    # 压缩失败，使用原始 prompt
                    self.session.effective_prompt = user_prompt
            else:
                # 不需要压缩
                self.session.compressed_prompt = None
                self.session.effective_prompt = user_prompt

            # 查询风格识别（使用原始 prompt）
            style_extractor = QueryStyleRecognition(self.llm, self.prompts, self.enable_logging)
            style_result = style_extractor.execute(user_prompt, self.language)
            if style_result:
                self.session.set_style(style_result)

            # 根节点建立（使用原始 prompt）
            root_establisher = RootNodeEstablishment(
                self.llm, self.prompts, self.evaluator, self.enable_logging,
                rag_service=self.rag_service
            )
            root, core_angle, core_structure = root_establisher.execute(
                user_prompt, total_words, style_result, self.session,
                sample_id=sample_id
            )

            # 注册根节点
            self.session.register_node(root)
            self.session.root_id = root.node_id

            # 全局摘要提取（使用 effective_prompt）
            style_context = self.session.get_style_context() if hasattr(self.session, 'get_style_context') else ""
            abstract_extractor = GlobalAbstract(self.llm, self.prompts, self.enable_logging)
            global_abstract = abstract_extractor.execute(core_angle, core_structure, total_words, style_context)
            self.session.global_abstract = global_abstract

            # DFS 节点扩展（使用 effective_prompt）
            checkpoint = ChapterLevelValidation(self.llm, self.prompts, self.enable_logging, self.evaluator)
            dfs_engine = DFSExpansionEngine(
                self.llm, self.prompts, self.evaluator,
                self.node_ops, self.tree_ops, self.enable_logging,
                checkpoint=checkpoint,
                rag_service=self.rag_service,
                max_width=self.max_width or MAX_WIDTH
            )
            root = dfs_engine.execute(self.session.get_effective_prompt(), root, self.session, sample_id)

            # 描述优化（在字数分配前）
            desc_optimizer = DescriptionOptimize(
                self.llm, self.prompts, self.enable_logging
            )
            root = desc_optimizer.execute(root, self.session)

            # BFS 字数分配
            allocator = WordCountAllocation(self.llm, self.prompts, self.node_ops, self.enable_logging)
            allocator.execute(self.session)

            # 全局校验与修复
            validator = GlobalValidation(self.llm, self.prompts, self.enable_logging)
            validator.execute(self.session)

            # 格式化输出
            formatter = Format(self.enable_logging)
            output = formatter.execute(self.session)

            if self.enable_logging:
                _print_and_flush("\n" + "=" * 70)
                _print_and_flush("Plan generation complete!")
                _print_and_flush("=" * 70)

            return output

        except Exception as e:
            if self.enable_logging:
                _print_and_flush(f"\n[Error] {e}")
                import traceback
                traceback.print_exc()
                _flush_stdout()
            raise

        finally:
            # 内存回收
            if self.session:
                self.session.clear()

    def get_stats(self) -> Dict[str, Any]:
        """获取调用统计"""
        return self.llm.get_stats()


class SyncISRPPlanner:
    """ISRP 同步包装器"""

    def __init__(
        self,
        client,
        max_depth: int = MAX_DEPTH,
        max_iterations: int = MAX_ITERATIONS,
        enable_logging: bool = True,
        language: str = None,
        persist_file: str = None,
        evaluator: 'BaseEvaluator' = None,
        rag_service = None,  # RAG 服务
        max_width: int = None  # 生成宽度上限
    ):
        self.planner = ISRPPlanner(
            client=client,
            max_depth=max_depth,
            max_iterations=max_iterations,
            enable_logging=enable_logging,
            language=language,
            persist_file=persist_file,
            evaluator=evaluator,
            rag_service=rag_service,  # 传递 RAG 服务
            max_width=max_width,
        )

    def generate_plan(self, user_prompt: str, total_words: int = 10000, sample_id: str = None) -> str:
        """生成大纲（同步接口）"""
        return asyncio.run(self.planner.generate_plan(user_prompt, total_words, sample_id))

    @property
    def llm(self):
        """LLM 客户端代理"""
        return self.planner.llm

    def get_stats(self) -> Dict[str, Any]:
        """获取 LLM 调用统计"""
        return self.planner.get_stats()


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ISRP Planner Test")
    print("=" * 70)
    print("\nNote: This test requires a valid API client configuration.")
    print("Run with: python run_plan.py")
    print("=" * 70)
