"""
机制技能系统 —— 万物皆可技能

核心理念：任何能返回文本的东西就是一个技能。
系统不关心技能是如何产生文本的，只关心文本本身。
"""

import os
import json
import subprocess
import importlib.util
from typing import List, Dict, Callable, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


# ── 触发时机枚举 ──────────────────────────────────────────────
class Trigger(Enum):
    ROUND_START = "round_start"
    ROUND_END = "round_end"
    BEFORE_SPEECH = "before_speech"
    AFTER_SPEECH = "after_speech"
    ON_VOTE = "on_vote"
    ON_CHALLENGE = "on_challenge"
    ON_ESSENCE_ADDED = "on_essence_added"
    ON_CONSENSUS = "on_consensus"
    ON_STALL = "on_stall"
    ON_DEBATE = "on_debate"
    ON_ELIMINATION = "on_elimination"
    EVERY_SPEECH = "every_speech"


# ── 安全格式化 ────────────────────────────────────────────────
class _SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


# ── 技能定义 ──────────────────────────────────────────────────
@dataclass
class MechanismSkill:
    """
    一个技能。

    核心契约：
    - execute() 永远返回文本 ({"text": "..."})
    - 系统只解析文本，不关心技能如何产生文本
    - skill_type 是提示"如何产生文本"，不是限制

    skill_type 可选值（仅提示作用，不强制）：
      code     → effect 是 Python 代码，返回文本
      text     → effect 就是文本本身
      template → effect 是 {var} 模板，格式化后返回
      llm      → effect 是提示词，发 LLM 返回文本
      shell    → effect 是 shell 命令，取 stdout
      (任何其他值) → effect 原样返回文本
    """
    name: str
    description: str
    trigger: str
    condition: str = "True"
    effect: str = ""
    skill_type: str = "text"
    enabled: bool = True
    priority: int = 0
    cooldown: int = 0
    uses_per_game: int = -1
    category: str = "默认"
    author: str = "系统"
    version: str = "1.0"

    # 运行时状态
    _uses_remaining: int = -1
    _last_triggered_round: int = -1
    _effect_func: Optional[Callable] = None
    _condition_func: Optional[Callable] = None

    def __post_init__(self):
        if self._uses_remaining < 0:
            self._uses_remaining = self.uses_per_game if self.uses_per_game > 0 else -1

    def check_condition(self, state: dict) -> bool:
        if self._condition_func:
            try:
                return self._condition_func(state)
            except Exception:
                return False
        try:
            return bool(eval(self.condition, {"__builtins__": {}}, state))
        except Exception:
            return False

    def execute(self, state: dict, engine: 'MechanismEngine') -> Dict:
        """
        执行技能，永远返回包含文本的字典。
        系统只关心返回的文本，不关心 execute 内部怎么做的。
        """
        # 1. 如果有绑定的 Python 函数，直接调用
        if self._effect_func:
            try:
                result = self._effect_func(state, engine)
                if isinstance(result, dict):
                    return result
                return {"text": str(result) if result is not None else ""}
            except Exception as e:
                return {"text": "", "error": str(e)}

        # 2. 根据 skill_type 生成文本
        text = ""
        st = self.skill_type

        if st == "code":
            try:
                local_vars = dict(state, engine=engine, text="")
                exec(self.effect, {"__builtins__": {}}, local_vars)
                text = str(local_vars.get("result") or local_vars.get("text", ""))
            except Exception as e:
                return {"text": "", "error": str(e)}

        elif st == "template" or st == "llm":
            try:
                text = self.effect.format_map(_SafeDict(state))
            except Exception as e:
                return {"text": "", "error": str(e)}
            if st == "llm":
                try:
                    client = engine.get_llm_client()
                    model = engine.get_param("llm_model", "deepseek-v4-flash")
                    if client is None:
                        return {"text": text, "error": "LLM client not available"}
                    response, _ = client.chat(
                        [{"role": "user", "content": text}],
                        model=model, thinking="disabled",
                        caller=f"skill:{self.name}",
                        show_reasoning=False, show_answer=False,
                    )
                    text = response.strip() if response else ""
                except Exception as e:
                    return {"text": text, "error": str(e)}

        elif st == "shell":
            try:
                result = subprocess.run(
                    self.effect, shell=True, capture_output=True,
                    text=True, timeout=10, encoding="utf-8",
                )
                text = result.stdout.strip()
            except Exception as e:
                return {"text": "", "error": str(e)}

        else:
            # text 或其他未识别的类型：effect 就是文本
            text = self.effect

        return {"text": text}

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "trigger": self.trigger,
            "condition": self.condition,
            "effect": self.effect,
            "skill_type": self.skill_type,
            "enabled": self.enabled,
            "priority": self.priority,
            "cooldown": self.cooldown,
            "uses_per_game": self.uses_per_game,
            "category": self.category,
            "author": self.author,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'MechanismSkill':
        return cls(**{k: v for k, v in data.items()
                      if k in cls.__dataclass_fields__})


# ── 技能模板 ──────────────────────────────────────────────────
SKILL_TEMPLATE = '''# 技能名称: {name}
# 描述: {description}
# 分类: {category}  |  作者: {author}  |  版本: {version}

# 触发时机（必填）
trigger: "{trigger}"

# 技能类型（提示系统如何产生文本，非强制）
#   code     → effect 是 Python 代码
#   text     → effect 就是文本本身（默认）
#   template → effect 是 {{var}} 模板
#   llm      → effect 是提示词，发送 LLM
#   shell    → effect 是 shell 命令
skill_type: "{skill_type}"

# 条件表达式（Python 布尔表达式）
condition: "{condition}"

# 效果内容（技能产生的文本来源）
effect: |
{effect_indent}

enabled: true
priority: {priority}
cooldown: {cooldown}
uses_per_game: {uses_per_game}
'''


# ── 技能引擎 ──────────────────────────────────────────────────
class MechanismEngine:
    """技能引擎——管理所有技能，执行时永远返回文本"""

    def __init__(self):
        self.skills: List[MechanismSkill] = []
        self._round_count: int = 0
        self._logs: List[str] = []
        self._pending_notifications: List[str] = []
        self._param_overrides: Dict[str, Any] = {}
        self._trigger_history: Dict[str, int] = {}
        self._skill_trigger_count: Dict[str, int] = {}
        self._skill_success_count: Dict[str, int] = {}
        self._skill_error_count: Dict[str, int] = {}
        self._llm_client = None

    # ── LLM 客户端 ──

    def set_llm_client(self, client) -> None:
        self._llm_client = client

    def get_llm_client(self):
        return self._llm_client

    # ── 技能注册 ──

    def register_skill(self, skill: MechanismSkill) -> None:
        skill.__post_init__()
        self.skills.append(skill)
        self.skills.sort(key=lambda s: s.priority, reverse=True)
        self._logs.append(f"技能注册: {skill.name} ({skill.trigger})")

    def register_skill_from_dict(self, skill_dict: dict) -> Optional[MechanismSkill]:
        try:
            skill = MechanismSkill.from_dict(skill_dict)
            self.register_skill(skill)
            self._logs.append(f"AI技能注册: {skill.name}")
            return skill
        except Exception as e:
            self._logs.append(f"技能注册失败: {e}")
            return None

    def register_skill_from_json(self, json_str: str) -> Optional[MechanismSkill]:
        try:
            data = json.loads(json_str)
            return self.register_skill_from_dict(data)
        except Exception as e:
            self._logs.append(f"JSON解析失败: {e}")
            return None

    def save_ai_skill(self, skill: MechanismSkill, directory: str = "skills/ai_generated") -> str:
        os.makedirs(directory, exist_ok=True)
        fname = f"{skill.name}.json"
        fpath = os.path.join(directory, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(skill.to_dict(), f, ensure_ascii=False, indent=2)
        self._logs.append(f"AI技能已保存: {fpath}")
        return fpath

    def remove_skill(self, name: str) -> bool:
        for i, s in enumerate(self.skills):
            if s.name == name:
                self.skills.pop(i)
                return True
        return False

    def get_skill(self, name: str) -> Optional[MechanismSkill]:
        for s in self.skills:
            if s.name == name:
                return s
        return None

    def toggle_skill(self, name: str, enabled: bool = None) -> bool:
        skill = self.get_skill(name)
        if skill:
            skill.enabled = not skill.enabled if enabled is None else enabled
            return True
        return False

    # ── 查询 ──

    def get_skills_by_category(self, category: str) -> List[MechanismSkill]:
        return [s for s in self.skills if s.category == category]

    def get_skills_by_trigger(self, trigger: str) -> List[MechanismSkill]:
        return [s for s in self.skills if s.trigger == trigger and s.enabled]

    def get_all_skills(self) -> List[MechanismSkill]:
        return list(self.skills)

    def get_categories(self) -> List[str]:
        cats = set()
        for s in self.skills:
            if s.category:
                cats.add(s.category)
        return sorted(cats)

    def search_skills(self, keyword: str) -> List[MechanismSkill]:
        kw = keyword.lower()
        return [s for s in self.skills
                if kw in s.name.lower()
                or kw in s.description.lower()
                or kw in s.trigger.lower()
                or kw in s.category.lower()
                or kw in s.skill_type.lower()]

    def get_skill_statistics(self) -> Dict:
        return {
            "total_skills": len(self.skills),
            "enabled_skills": sum(1 for s in self.skills if s.enabled),
            "trigger_counts": dict(self._skill_trigger_count),
            "success_counts": dict(self._skill_success_count),
            "error_counts": dict(self._skill_error_count),
            "event_trigger_counts": dict(self._trigger_history),
            "categories": self.get_categories(),
        }

    def get_skill_detail(self, name: str) -> Optional[Dict]:
        skill = self.get_skill(name)
        if not skill:
            return None
        return {
            "name": skill.name,
            "description": skill.description,
            "trigger": skill.trigger,
            "skill_type": skill.skill_type,
            "condition": skill.condition,
            "effect": skill.effect[:200] + ("..." if len(skill.effect) > 200 else ""),
            "enabled": skill.enabled,
            "priority": skill.priority,
            "cooldown": skill.cooldown,
            "uses_per_game": skill.uses_per_game,
            "category": skill.category,
            "author": skill.author,
            "version": skill.version,
            "trigger_count": self._skill_trigger_count.get(skill.name, 0),
            "success_count": self._skill_success_count.get(skill.name, 0),
            "error_count": self._skill_error_count.get(skill.name, 0),
        }

    # ── 核心触发 ──

    def trigger(self, event: str, game_state: dict) -> List[Dict]:
        """
        触发事件，执行匹配的技能。
        每个技能返回 {"text": "...", "skill_name": "...", ...}
        """
        self._trigger_history[event] = self._trigger_history.get(event, 0) + 1
        results = []

        for skill in self.get_skills_by_trigger(event):
            if skill.cooldown > 0 and skill._last_triggered_round > 0:
                if self._round_count - skill._last_triggered_round < skill.cooldown:
                    continue
            if skill._uses_remaining == 0:
                continue

            state = dict(game_state, engine=self)
            if not skill.check_condition(state):
                continue

            result = skill.execute(state, self)
            skill._last_triggered_round = self._round_count
            if skill._uses_remaining > 0:
                skill._uses_remaining -= 1

            result["skill_name"] = skill.name
            result["trigger"] = event
            result["skill_type"] = skill.skill_type
            results.append(result)

            self._skill_trigger_count[skill.name] = self._skill_trigger_count.get(skill.name, 0) + 1
            if "error" not in result or not result["error"]:
                self._skill_success_count[skill.name] = self._skill_success_count.get(skill.name, 0) + 1
            else:
                self._skill_error_count[skill.name] = self._skill_error_count.get(skill.name, 0) + 1

            self._logs.append(f"[{event}] {skill.name} → {result.get('text','')[:60]}")

        return results

    def advance_round(self) -> None:
        self._round_count += 1

    # ── 辅助方法 ──

    def add_log(self, msg: str) -> None:
        self._logs.append(msg)

    def notify(self, msg: str) -> None:
        self._pending_notifications.append(msg)

    def modify_param(self, key: str, value: Any) -> None:
        self._param_overrides[key] = value

    def get_param(self, key: str, default: Any = None) -> Any:
        return self._param_overrides.get(key, default)

    def consume_notifications(self) -> List[str]:
        notes = list(self._pending_notifications)
        self._pending_notifications.clear()
        return notes

    # ── 内置技能 ──

    def add_builtin_skills(self) -> None:
        builtins = [
            MechanismSkill(
                name="投票机制", description="每轮结束后对精华进行投票，高票精华获得加分",
                trigger=Trigger.ROUND_END.value,
                condition="game_state.get('enable_vote', True) and len(game_state.get('vote_essences', [])) > 0",
                effect="", priority=10, category="核心机制",
            ),
            MechanismSkill(
                name="辩论机制", description="对有争议的精华进行深入辩论",
                trigger=Trigger.ON_CHALLENGE.value,
                condition="game_state.get('enable_debate', True)",
                effect="", priority=10, category="核心机制",
            ),
            MechanismSkill(
                name="赏金系统", description="每轮玩家可秘密悬赏一名存活玩家",
                trigger=Trigger.ROUND_START.value,
                condition="game_state.get('enable_bounty', False) and len(game_state.get('alive_players', [])) >= 3",
                effect="赏金系统已激活", priority=8, category="扩展机制",
                skill_type="text",
            ),
            MechanismSkill(
                name="保护令牌", description="每位玩家初始拥有1个保护令牌，可抵消1次曝光度增加",
                trigger=Trigger.ON_ELIMINATION.value,
                condition="game_state.get('enable_protection', False)",
                effect="", priority=8, category="扩展机制",
            ),
            MechanismSkill(
                name="死亡赌注", description="被挑战时可发起一次死亡赌注",
                trigger=Trigger.ON_CHALLENGE.value,
                condition="game_state.get('enable_death_gamble', False)",
                effect="", priority=8, category="扩展机制",
            ),
            MechanismSkill(
                name="谣言工厂", description="每2轮随机公开1-2条匿名谣言",
                trigger=Trigger.ROUND_END.value,
                condition="game_state.get('enable_rumor', False) and game_state.get('round_count', 0) % 2 == 0",
                effect="谣言工厂已触发", priority=6, category="扩展机制",
                skill_type="text",
            ),
            MechanismSkill(
                name="闲聊", description="专家发言前可进行简短闲聊",
                trigger=Trigger.BEFORE_SPEECH.value,
                condition="game_state.get('enable_banter', False)",
                effect="", priority=5, category="社交机制",
            ),
            MechanismSkill(
                name="侦察", description="专家可主动侦察其他专家的专业领域和立场",
                trigger=Trigger.ROUND_START.value,
                condition="game_state.get('enable_scout', False) and game_state.get('round_count', 0) >= 2",
                effect="侦察阶段：专家们正在互相了解专业背景",
                priority=5, category="社交机制", skill_type="text",
            ),
            MechanismSkill(
                name="桌边谈", description="讨论期间可进行非正式交流",
                trigger=Trigger.EVERY_SPEECH.value,
                condition="game_state.get('enable_table_talk', False)",
                effect="", priority=5, category="社交机制",
            ),
            MechanismSkill(
                name="自适应停止建议", description="当共识度达标或僵持过久时，自动建议用户结束讨论",
                trigger=Trigger.ROUND_END.value,
                condition="game_state.get('enable_auto_stop', True)",
                effect="", priority=3, category="系统",
            ),
            MechanismSkill(
                name="实时精华反馈", description="发言过程中关键洞察即时入池",
                trigger=Trigger.AFTER_SPEECH.value,
                condition="game_state.get('enable_real_time_feedback', True)",
                effect="", priority=3, category="系统",
            ),
        ]
        for skill in builtins:
            self.register_skill(skill)

    # ── 从文件加载（万物皆可加载） ──

    def load_skills_from_dir(self, directory: str) -> int:
        """
        从目录加载技能。
        任何文件都可以是技能——系统只关心文件的文本内容。
        """
        if not os.path.isdir(directory):
            return 0

        count = 0
        for fname in sorted(os.listdir(directory)):
            fpath = os.path.join(directory, fname)

            # 跳过隐藏文件和目录
            if fname.startswith("_") or fname.startswith(".") or os.path.isdir(fpath):
                continue

            try:
                state = self._read_file_as_skill(fpath, fname)
                if state:
                    self.register_skill(state)
                    count += 1
            except Exception as e:
                self._logs.append(f"加载技能失败 {fname}: {e}")

        return count

    def _parse_comment_header(self, content: str) -> Dict[str, str]:
        """解析文件头部注释中的 # key: value 声明"""
        meta = {}
        for line in content.split("\n"):
            stripped = line.strip()
            if not stripped.startswith("#"):
                break  # 遇到非注释行就停止解析
            # 去掉 # 前缀
            rest = stripped[1:].strip()
            if ":" in rest:
                key, _, val = rest.partition(":")
                key = key.strip().lower()
                val = val.strip()
                if val and key in ("trigger", "condition", "name", "description",
                                   "priority", "cooldown", "uses_per_game",
                                   "category", "author", "version", "skill_type"):
                    meta[key] = val
        return meta

    def _read_file_as_skill(self, fpath: str, fname: str) -> Optional[MechanismSkill]:
        """读取任意文件并注册为技能"""
        ext = os.path.splitext(fname)[1].lower()

        # ── .json：解析为技能定义 ──
        if ext == ".json":
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return None
            return MechanismSkill.from_dict(data)

        # ── .py：支持 # trigger: round_end 注释声明 ──
        if ext == ".py":
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()

            meta = self._parse_comment_header(content)

            # 如果文件头包含 # trigger: 声明，就用注释解析方式
            if "trigger" in meta:
                try:
                    priority = int(meta.get("priority", 0))
                except ValueError:
                    priority = 0
                try:
                    cooldown = int(meta.get("cooldown", 0))
                except ValueError:
                    cooldown = 0
                try:
                    uses = int(meta.get("uses_per_game", -1))
                except ValueError:
                    uses = -1

                return MechanismSkill(
                    name=meta.get("name", fname[:-3]),
                    description=meta.get("description", f"Python 技能 (来自 {fname})"),
                    trigger=meta["trigger"],
                    condition=meta.get("condition", "True"),
                    effect=content,
                    skill_type=meta.get("skill_type", "code"),
                    enabled=True,
                    priority=priority,
                    cooldown=cooldown,
                    uses_per_game=uses,
                    category=meta.get("category", "Python 技能"),
                    author=meta.get("author", "文件加载"),
                    version=meta.get("version", "1.0"),
                )

            # 没有 # trigger 声明：尝试 exec 获取 create_skill()
            spec = importlib.util.spec_from_file_location(fname[:-3], fpath)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, "create_skill"):
                    skill = mod.create_skill()
                    if isinstance(skill, MechanismSkill):
                        return skill
                    if isinstance(skill, list):
                        return None
            # 没有 create_skill 函数，把整个文件当 code 技能
            return MechanismSkill(
                name=fname[:-3],
                description=f"Python 技能 (来自 {fname})",
                trigger="round_end",
                effect=content,
                skill_type="code",
                category="Python 技能",
                author="文件加载",
            )

        # ── 其他任何文件（.txt, .md, .yaml, .csv, .html, etc.）──
        # 只要是文本文件，就作为 text 技能加载
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if content:
                return MechanismSkill(
                    name=fname.replace(".", "_"),
                    description=f"文本技能 (来自 {fname})",
                    trigger="round_end",
                    effect=content,
                    skill_type="text",
                    category=f"文件来源 ({ext[1:] if ext else '未知'})",
                    author="文件加载",
                )
        except (UnicodeDecodeError, Exception):
            pass  # 二进制文件跳过

        return None

    def create_skill_file(self, directory: str, name: str, **kwargs) -> str:
        """创建技能模板文件"""
        if not os.path.isdir(directory):
            os.makedirs(directory, exist_ok=True)

        fpath = os.path.join(directory, f"{name}.json")
        trigger_choices = ", ".join(t.value for t in Trigger)

        effect = kwargs.get("effect", "# 在此编写效果内容")
        effect_lines = effect.split("\n")
        effect_indented = "\n".join(f"  {line}" for line in effect_lines)

        content = SKILL_TEMPLATE.format(
            name=kwargs.get("name", name),
            description=kwargs.get("description", "自定义技能"),
            category=kwargs.get("category", "自定义"),
            author=kwargs.get("author", "用户"),
            version=kwargs.get("version", "1.0"),
            skill_type=kwargs.get("skill_type", "text"),
            trigger_choices=trigger_choices,
            trigger=kwargs.get("trigger", "round_end"),
            condition=kwargs.get("condition", "True"),
            effect_indent=effect_indented,
            priority=kwargs.get("priority", 0),
            cooldown=kwargs.get("cooldown", 0),
            uses_per_game=kwargs.get("uses_per_game", -1),
        )

        with open(fpath, "w", encoding="utf-8") as f:
            f.write(content)

        return fpath

    # ── 序列化 ──

    def to_dict(self) -> dict:
        return {
            "round_count": self._round_count,
            "skills": [s.to_dict() for s in self.skills],
            "param_overrides": self._param_overrides,
            "logs": self._logs[-100:],
            "skill_trigger_count": dict(self._skill_trigger_count),
            "skill_success_count": dict(self._skill_success_count),
            "skill_error_count": dict(self._skill_error_count),
            "trigger_history": dict(self._trigger_history),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'MechanismEngine':
        engine = cls()
        engine._round_count = data.get("round_count", 0)
        for skill_data in data.get("skills", []):
            engine.register_skill(MechanismSkill.from_dict(skill_data))
        engine._param_overrides = data.get("param_overrides", {})
        engine._logs = data.get("logs", [])
        engine._skill_trigger_count = data.get("skill_trigger_count", {})
        engine._skill_success_count = data.get("skill_success_count", {})
        engine._skill_error_count = data.get("skill_error_count", {})
        engine._trigger_history = data.get("trigger_history", {})
        return engine