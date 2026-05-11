"""
Phase III — 格式化输出

将大纲树转换为标准化的段落列表：
  - 验证树结构完整性
  - 按 DFS 遍历顺序展开叶子节点
  - 生成 AgentWrite Write 模块兼容的段落格式
每个段落包含编号、标题、描述与分配字数。

"""
import sys
from ..schemas import validate_paragraph_format
from ..utils.tree_ops import TreeOperations

# 导入实时输出辅助函数
from . import print_and_flush, flush_stdout


class Format:
    """Format"""

    def __init__(self, enable_logging: bool = True):
        """
        初始化格式化器

        Args:
            enable_logging: 是否启用日志
        """
        self.enable_logging = enable_logging

    def execute(self, session) -> str:
        """
        执行格式化输出

        Args:
            session: 会话实例（PlanSession）

        Returns:
            LongWriter 标准格式的大纲字符串
        """
        if self.enable_logging:
            print_and_flush("\n[Format] Format Output")
            print_and_flush("=" * 60)

        # 验证树完整性
        issues = TreeOperations.validate_integrity(session)
        if issues:
            if self.enable_logging:
                print_and_flush(f"  [Warning] Tree integrity issues detected: {len(issues)}")
            TreeOperations.repair_integrity(session, issues)

            # 重新验证
            remaining_issues = TreeOperations.validate_integrity(session)
            if remaining_issues:
                if self.enable_logging:
                    print_and_flush(f"  [Error] Cannot repair {len(remaining_issues)} issues: {remaining_issues[:3]}...")

        # 获取叶子节点
        leaves = session.get_all_leaf_nodes()

        # 格式化段落
        lines = []
        for i, leaf in enumerate(leaves, 1):
            # 获取完整路径上下文（已包含编号）
            path_context = leaf.get_path_context(session.node_registry)

            # 去除标题中已有的编号前缀，避免重复
            import re
            clean_title = re.sub(r'^[\d.]+\s*', '', leaf.title).strip()

            description = leaf.description if leaf.description else ""
            # 清理换行符，确保单行格式
            clean_title = clean_title.replace('\n', ' ').replace('\r', ' ')
            description = description.replace('\n', ' ').replace('\r', ' ')
            # 压缩连续空格
            while '  ' in clean_title:
                clean_title = clean_title.replace('  ', ' ')
            while '  ' in description:
                description = description.replace('  ', ' ')

            # 构建段落行
            main_point = f"{path_context} - {description}".strip()
            line = f"[Paragraph {i} - Main Point : {main_point} - Word Count : {leaf.allocated_words}]"
            lines.append(line)

        output = "\n".join(lines)

        # 验证格式
        is_valid = validate_paragraph_format(output)

        if self.enable_logging:
            print_and_flush(f"  Generated {len(lines)} paragraphs")
            print_and_flush(f"  Format valid: {is_valid}")

        return output