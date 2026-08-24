"""
多模态输入/输出扩展 —— 语音、文件附件、媒体显示

功能：
- 语音输出（TTS）：将讨论内容朗读出来
- 语音输入（STT）：通过语音输入命令/提问（含文本备选）
- 文件附件：将图片/文档/数据附加到讨论中作为上下文
- 媒体显示：在GUI中展示附件内容
"""

import os
# 在导入 pygame 前屏蔽其欢迎信息和第三方库弃用警告
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", message="Hello from the pygame community")

import time
import threading
import datetime
import json
import urllib.request
import urllib.error
import tempfile
import subprocess
from typing import List, Optional, Dict
from enum import Enum

try:
    import requests as _requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from .prompts_b64 import _get_b64_prompt

from PySide6.QtCore import Qt, Signal, QObject
from PySide6.QtGui import QPixmap, QFont
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QWidget, QListWidget, QListWidgetItem,
    QFileDialog, QMessageBox, QCheckBox, QGroupBox,
    QTabWidget, QSlider, QScrollArea, QSizePolicy,
    QSplitter,
)


# ── 语音输出（TTS）──

class TTSProvider(Enum):
    KOKORO = "Kokoro (本地AI)"
    EDGE_TTS = "edge-tts (微软)"
    PYTTSX3 = "pyttsx3 (本地)"
    WINSOUND = "winsound (蜂鸣)"
    ONLINE_OPENAI = "在线 TTS (xh.v1api.cc)"
    NONE = "不可用"


class TextToSpeech:
    """
    文本转语音输出模块。

    支持：
    - 离线 TTS（pyttsx3）
    - 蜂鸣（winsound 备选）
    - 在线 TTS（OpenAI 兼容接口，通过 xh.v1api.cc 代理）
    """

    def __init__(self):
        self._engine = None
        self._provider = TTSProvider.NONE
        self._enabled = True  # 默认启用（由 _init_engine 决定具体引擎）
        self._rate = 180
        self._volume = 0.8
        self._voice_id = None
        self._lock = threading.Lock()
        self._audio_started = threading.Event()  # 音频开始播放时触发

        # Kokoro TTS（本地AI语音合成）
        self._kokoro_pipeline = None
        self._kokoro_voice = "zm_yunyang"  # 男声，浑厚有力

        # 在线 TTS 配置
        self._online_api_key = ""
        self._online_base_url = _get_b64_prompt("TTS_BASE_URL")
        self._online_voice = "alloy"
        self._online_model = "tts-1"

        # edge-tts 配置
        self._edge_voice = "zh-CN-YunyangNeural"  # 男声（新闻腔，较机械）
        self._edge_rate = "+0%"  # 正常语速

        self._init_engine()

    def configure_online(self, api_key: str, base_url: str = "",
                         voice: str = "alloy", model: str = "tts-1"):
        """配置在线 TTS 参数（只保存配置，不切换引擎）"""
        self._online_api_key = api_key
        self._online_base_url = base_url.rstrip("/").rstrip("/v1") if base_url else _get_b64_prompt("TTS_BASE_URL")
        self._online_voice = voice
        self._online_model = model

    def switch_to_online(self):
        """切换到在线 TTS 引擎"""
        if self._online_api_key and self._online_base_url:
            self._provider = TTSProvider.ONLINE_OPENAI
            self._enabled = True

    def switch_to_local(self):
        """切换到本地 TTS 引擎"""
        self._init_engine()

    def switch_to_edge_tts(self):
        """切换到 edge-tts 引擎"""
        try:
            import edge_tts
            self._provider = TTSProvider.EDGE_TTS
            self._enabled = True
        except ImportError:
            print("  🔊 edge-tts 未安装，请执行: pip install edge-tts")
            self._init_engine()

    def _init_engine(self):
        """初始化 TTS 引擎（Kokoro → pyttsx3 → edge-tts → winsound → NONE）"""
        # 1) 优先尝试 Kokoro（本地AI语音合成，无需网络，质量高）
        try:
            import os as _os
            _os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            from kokoro import KPipeline
            self._kokoro_pipeline = KPipeline(lang_code='z', repo_id='hexgrad/Kokoro-82M')
            self._provider = TTSProvider.KOKORO
            return
        except Exception:
            self._kokoro_pipeline = None

        # 2) 尝试 pyttsx3（SAPI5 本地语音，无需网络，稳定可靠）
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self._rate)
            self._engine.setProperty("volume", self._volume)
            voices = self._engine.getProperty("voices")
            # 优先选择中文语音，否则用默认的 Microsoft David（机械男声）
            for v in voices:
                if "zh" in v.id.lower() or "chinese" in v.name.lower():
                    self._voice_id = v.id
                    break
            if self._voice_id:
                self._engine.setProperty("voice", self._voice_id)
            self._provider = TTSProvider.PYTTSX3
            return
        except Exception:
            pass

        # 3) 尝试 edge-tts（微软在线语音，需网络）
        try:
            import edge_tts
            _ = edge_tts.Communicate
            self._provider = TTSProvider.EDGE_TTS
            return
        except Exception:
            pass

        # 4) 蜂鸣回退
        try:
            import winsound
            self._provider = TTSProvider.WINSOUND
        except Exception:
            self._provider = TTSProvider.NONE

    @property
    def available(self) -> bool:
        return self._provider != TTSProvider.NONE

    @property
    def enabled(self) -> bool:
        return self._enabled and self.available

    @enabled.setter
    def enabled(self, val: bool):
        self._enabled = val

    @property
    def provider_name(self) -> str:
        return self._provider.value

    def speak(self, text: str) -> bool:
        """
        朗读文本。在后台线程中执行。
        返回是否成功启动。
        """
        if not self.enabled or not text.strip():
            return False

        threading.Thread(target=self._speak_internal,
                         args=(text,), daemon=True).start()
        return True

    def speak_sync(self, text: str, show_debug: bool = True) -> bool:
        """
        同步朗读（阻塞直到播放完成），用于测试。
        show_debug=True 时输出每个步骤的状态。
        """
        if not text.strip():
            return False
        if self._provider == TTSProvider.NONE:
            if show_debug:
                print(f"  ✗ 无可用语音引擎（_provider = NONE）")
            return False
        if show_debug:
            print(f"  🔊 开始朗读 ({len(text)} 字)")
            print(f"     引擎: {self.provider_name}")
            if self._provider == TTSProvider.ONLINE_OPENAI:
                print(f"     接口: {self._online_base_url}")
            elif self._provider == TTSProvider.EDGE_TTS:
                print(f"     语音: {self._edge_voice}, 语速: {self._edge_rate}")
            else:
                print(f"     语速: {self._rate}, 音量: {self._volume}")
        self._speak_internal(text, show_debug)
        if show_debug:
            print(f"  ✓ 朗读完成")
        return True

    def speak_async_start(self, text: str, timeout: float = 15.0) -> bool:
        """启动 TTS 朗读，等待音频真正开始播放后返回"""
        if not self.enabled or not text.strip():
            return False
        self._audio_started.clear()
        threading.Thread(target=self._speak_internal,
                         args=(text,), daemon=True).start()
        self._audio_started.wait(timeout=timeout)
        return True

    def _speak_internal(self, text: str, show_debug: bool = False):
        with self._lock:
            try:
                if self._provider == TTSProvider.KOKORO and self._kokoro_pipeline:
                    self._audio_started.set()
                    gen = self._kokoro_pipeline(text, voice=self._kokoro_voice)
                    import sounddevice as sd
                    for result in gen:
                        tensor = result.output.audio
                        audio = tensor.detach().cpu().numpy()
                        if len(audio) > 0:
                            sd.play(audio, 24000)
                            sd.wait()
                elif self._provider == TTSProvider.EDGE_TTS:
                    # 使用 asyncio 运行 edge-tts
                    import asyncio
                    asyncio.run(self._speak_edge_tts(text, show_debug))
                elif self._provider == TTSProvider.PYTTSX3 and self._engine:
                    self._audio_started.set()
                    self._engine.say(text)
                    self._engine.runAndWait()
                elif self._provider == TTSProvider.WINSOUND:
                    import winsound
                    for _ in range(min(len(text) // 20, 5)):
                        winsound.Beep(800, 200)
                        time.sleep(0.1)
                elif self._provider == TTSProvider.ONLINE_OPENAI:
                    self._speak_online(text, show_debug)
            except Exception as e:
                print(f"  🔊 TTS异常: {e}")

    def _speak_online(self, text: str, show_debug: bool = False):
        """
        通过在线 API 合成语音并播放。
        将长文本自动切分成短句（每句约80字）分批请求，
        避免单次请求过大导致超时。
        全部失败时回退到蜂鸣。
        """
        if not self._online_api_key or not self._online_base_url:
            print(f"  🔊 TTS: 未配置 API Key 或接口地址")
            self._fallback_beep(text)
            return

        # 将文本切分成短句（按标点或固定长度分割）
        chunks = self._split_tts_text(text, max_chars=80)

        # 尝试的模型列表（去重）
        models_to_try = list(dict.fromkeys([
            self._online_model,
            "tts-1-hd",
        ]))

        url = f"{self._online_base_url}/v1/audio/speech"
        headers = {
            "Authorization": f"Bearer {self._online_api_key}",
            "Content-Type": "application/json",
        }

        total_chunks = len(chunks)
        audio_parts = []

        for chunk_idx, chunk_text in enumerate(chunks):
            if show_debug:
                print(f"  → TTS 段落 {chunk_idx+1}/{total_chunks} ({len(chunk_text)}字)")

            chunk_audio = None
            last_error = ""
            for model in models_to_try:
                payload = {
                    "model": model,
                    "input": chunk_text,
                    "voice": self._online_voice,
                    "response_format": "mp3",
                }

                if show_debug or model != models_to_try[0]:
                    print(f"  → TTS 尝试模型: {model}")

                try:
                    t0 = time.time()
                    if HAS_REQUESTS:
                        resp = _requests.post(
                            url,
                            json=payload,
                            headers=headers,
                            timeout=(10, 30),  # 连接10s, 读取30s
                            stream=False,
                        )
                        if resp.status_code != 200:
                            raise Exception(f"HTTP {resp.status_code}: {resp.text[:80]}")
                        chunk_audio = resp.content
                    else:
                        data = json.dumps(payload).encode("utf-8")
                        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
                        with urllib.request.urlopen(req, timeout=30) as resp:
                            chunk_audio = resp.read()
                    elapsed = time.time() - t0
                    if show_debug:
                        print(f"  ✓ 响应 ({elapsed:.1f}s), {len(chunk_audio)} 字节")
                    break  # 当前段落成功
                except Exception as e:
                    last_error = str(e)
                    if show_debug or model != models_to_try[-1]:
                        print(f"  ⚠ [{model}] 失败: {last_error[:80]}")
                    continue

            if chunk_audio:
                audio_parts.append(chunk_audio)
            else:
                print(f"  🔊 TTS 段落 {chunk_idx+1} 合成失败: {last_error[:80]}")
                # 继续尝试后续段落

        if not audio_parts:
            print(f"  🔊 TTS 全部段落合成失败")
            self._fallback_beep(text)
            return

        # 合并所有音频段并播放
        self._play_audio_parts(audio_parts, show_debug)

    async def _speak_edge_tts(self, text: str, show_debug: bool = False):
        """
        通过 edge-tts（微软免费 TTS）合成语音并静默播放。
        自动将长文本切分为短句分批合成，避免超时。
        """
        import edge_tts
        import pygame
        import asyncio

        # 将文本切分成短句（按标点或固定长度分割）
        chunks = self._split_tts_text(text, max_chars=500)
        total_chunks = len(chunks)
        audio_parts = []

        for idx, chunk in enumerate(chunks):
            if show_debug:
                print(f"  → edge-tts 段落 {idx+1}/{total_chunks} ({len(chunk)}字)")

            # 应用语速调整
            rate = self._edge_rate

            try:
                communicate = edge_tts.Communicate(
                    chunk,
                    voice=self._edge_voice,
                    rate=rate,
                )
                fd, tmp_path = tempfile.mkstemp(suffix=".mp3")
                os.close(fd)
                t0 = time.time()
                await communicate.save(tmp_path)
                elapsed = time.time() - t0

                if show_debug:
                    print(f"  ✓ 合成成功 ({elapsed:.1f}s)")

                # 读取音频数据加入播放列表
                with open(tmp_path, "rb") as f:
                    audio_data = f.read()
                audio_parts.append(audio_data)

                # 立即播放（边合成边播，减少等待感）
                try:
                    if not pygame.mixer.get_init():
                        pygame.mixer.init()
                    pygame.mixer.music.load(tmp_path)
                    pygame.mixer.music.play()
                    self._audio_started.set()  # 音频已开始播放
                    while pygame.mixer.music.get_busy():
                        await asyncio.sleep(0.1)
                    pygame.mixer.music.unload()
                except Exception as play_err:
                    if show_debug:
                        print(f"  ⚠ 播放警告: {play_err}")

                # 清理临时文件
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

            except Exception as e:
                print(f"  🔊 edge-tts 段落 {idx+1} 失败: {e}")
                # 继续合成下一段

        if not audio_parts:
            print(f"  🔊 edge-tts 全部失败，回退蜂鸣")
            self._fallback_beep(text)
            return

        if show_debug:
            print(f"  ✓ edge-tts 朗读完成 ({len(audio_parts)} 段)")

    def _split_tts_text(self, text: str, max_chars: int = 80) -> List[str]:
        """将长文本按标点符号或固定长度切分成短句"""
        if len(text) <= max_chars:
            return [text]

        # 优先按标点分割
        import re
        # 按句号、问号、感叹号、分句、换行分割
        sentences = re.split(r'(?<=[。！？\n])\s*', text)
        chunks = []
        current = ""
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if len(current) + len(s) <= max_chars:
                current += s
            else:
                if current:
                    chunks.append(current)
                # 如果单句超长，按固定长度截断
                while len(s) > max_chars:
                    chunks.append(s[:max_chars])
                    s = s[max_chars:]
                current = s
        if current:
            chunks.append(current)
        return chunks if chunks else [text[:max_chars]]

    def _play_audio_parts(self, audio_parts: List[bytes], show_debug: bool = False):
        """将多个音频片段依次保存并播放"""
        import pygame
        tmp_paths = []
        try:
            for i, audio_data in enumerate(audio_parts):
                fd, tmp_path = tempfile.mkstemp(suffix=".mp3")
                os.close(fd)
                with open(tmp_path, "wb") as f:
                    f.write(audio_data)
                tmp_paths.append(tmp_path)

            for i, tmp_path in enumerate(tmp_paths):
                if show_debug and len(tmp_paths) > 1:
                    print(f"  → 播放段落 {i+1}/{len(tmp_paths)}")
                try:
                    if not pygame.mixer.get_init():
                        pygame.mixer.init()
                    pygame.mixer.music.load(tmp_path)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                finally:
                    try:
                        pygame.mixer.music.unload()
                    except Exception:
                        pass
            if show_debug:
                print(f"  ✓ 播放完成 ({len(audio_parts)} 段)")

        except Exception as e:
            print(f"  🔊 TTS 播放失败: {e}")
            if not show_debug:
                self._fallback_beep(" ".join(audio_parts) if audio_parts else "")
        finally:
            # 清理临时文件
            for p in tmp_paths:
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass

    def _play_audio_silent(self, path: str, show_debug: bool = False):
        """使用 pygame.mixer 静默播放 MP3，不显示任何窗口"""
        import pygame
        try:
            if not pygame.mixer.get_init():
                if show_debug:
                    print(f"  → 初始化 pygame.mixer...")
                pygame.mixer.init()
                if show_debug:
                    print(f"  ✓ mixer 已初始化")
            if show_debug:
                print(f"  → 加载音频...")
            pygame.mixer.music.load(path)
            if show_debug:
                print(f"  ✓ 加载成功，开始播放...")
            pygame.mixer.music.play()
            busy = True
            while busy:
                time.sleep(0.1)
                busy = pygame.mixer.music.get_busy()
            if show_debug:
                print(f"  ✓ 播放完成")
        finally:
            try:
                pygame.mixer.music.unload()
            except Exception:
                pass

    def _fallback_beep(self, text: str):
        """在线 TTS 失败时的蜂鸣回退"""
        try:
            import winsound
            for _ in range(min(len(text) // 20, 3)):
                winsound.Beep(800, 200)
                time.sleep(0.1)
        except Exception:
            pass

    def speak_async(self, text: str, callback=None):
        """
        异步朗读，可选回调。
        """
        def _run():
            self._speak_internal(text)
            if callback:
                callback()
        threading.Thread(target=_run, daemon=True).start()

    def set_rate(self, rate: int):
        """设置语速（50-300）"""
        self._rate = max(50, min(300, rate))
        if self._engine:
            try:
                self._engine.setProperty("rate", self._rate)
            except Exception:
                pass

    def set_volume(self, volume: float):
        """设置音量（0.0-1.0）"""
        self._volume = max(0.0, min(1.0, volume))
        if self._engine:
            try:
                self._engine.setProperty("volume", self._volume)
            except Exception:
                pass

    def stop(self):
        """停止朗读"""
        with self._lock:
            try:
                if self._engine:
                    self._engine.stop()
            except Exception:
                pass


# 全局TTS实例
_tts = TextToSpeech()


def get_tts() -> TextToSpeech:
    return _tts


# ── 语音输入（STT）──

class SpeechToText:
    """
    语音输入模块。

    由于环境缺少 speech_recognition 库，提供基于文本的备选输入。
    用户可以通过麦克风图标或快捷键切换到"语音输入模式"，
    实际以文本输入替代。
    """

    def __init__(self):
        self._listening = False

    @property
    def available(self) -> bool:
        return False  # 无 STT 库，标记为不可用

    def listen(self, timeout: float = 5.0) -> Optional[str]:
        """
        模拟语音输入 —— 实际返回 None，提示用户手动输入。
        真实场景下可替换为 speech_recognition。
        """
        return None

    def listen_async(self, callback, timeout: float = 5.0):
        """异步语音输入"""
        result = self.listen(timeout)
        if callback:
            callback(result)


# ── 文件附件系统 ──

class AttachmentType(Enum):
    IMAGE = "图片"
    TEXT = "文本"
    DATA = "数据文件"
    OTHER = "其他"


ATTACHMENT_DIR = "attachments"


class Attachment:
    """单个附件"""

    def __init__(self, file_path: str, description: str = ""):
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)
        self.description = description or self.file_name
        self.attached_at = datetime.datetime.now().isoformat()

        ext = os.path.splitext(file_path)[1].lower()
        if ext in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"):
            self.attach_type = AttachmentType.IMAGE
        elif ext in (".txt", ".md", ".py", ".json", ".csv", ".xml", ".yaml", ".yml"):
            self.attach_type = AttachmentType.TEXT
        elif ext in (".csv", ".xlsx", ".xls", ".json", ".xml"):
            self.attach_type = AttachmentType.DATA
        else:
            self.attach_type = AttachmentType.OTHER

        self._content_cache = None

    @property
    def type_name(self) -> str:
        return self.attach_type.value

    def read_text(self, max_len: int = 2000) -> str:
        """读取文本内容（用于给LLM提供上下文）"""
        if self._content_cache:
            return self._content_cache
        try:
            with open(self.file_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read(max_len)
            self._content_cache = text
            return text
        except Exception:
            return f"（无法读取文件: {self.file_name}）"

    def get_context_text(self) -> str:
        """生成附件上下文文本（供LLM使用）"""
        header = f"[附件: {self.file_name}]"
        if self.attach_type == AttachmentType.IMAGE:
            return f"{header} (图片文件，已保存到附件目录)"
        content = self.read_text(1000)
        return f"{header}\n{content}\n[附件结束]"

    def to_dict(self) -> Dict:
        return {
            "file_path": self.file_path,
            "file_name": self.file_name,
            "description": self.description,
            "attached_at": self.attached_at,
            "type": self.attach_type.value,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Attachment":
        att = cls(data["file_path"], data.get("description", ""))
        att.attached_at = data.get("attached_at", att.attached_at)
        return att


class AttachmentManager:
    """
    附件管理器。

    管理讨论中的文件附件，支持：
    - 添加/移除附件
    - 为LLM提供附件上下文
    - 附件持久化
    """

    def __init__(self):
        self.attachments: List[Attachment] = []
        if not os.path.exists(ATTACHMENT_DIR):
            os.makedirs(ATTACHMENT_DIR)

    def add(self, file_path: str, description: str = "") -> Optional[Attachment]:
        """添加附件"""
        if not os.path.exists(file_path):
            return None
        att = Attachment(file_path, description)
        self.attachments.append(att)
        return att

    def remove(self, index: int) -> bool:
        """移除附件"""
        if 0 <= index < len(self.attachments):
            self.attachments.pop(index)
            return True
        return False

    def clear(self):
        """清空附件"""
        self.attachments.clear()

    def get_context(self) -> str:
        """获取所有附件的上下文文本（供LLM使用）"""
        if not self.attachments:
            return ""
        parts = ["\n\n[已附加文件]"]
        for att in self.attachments:
            parts.append(att.get_context_text())
        parts.append("[附件结束]\n")
        return "\n".join(parts)

    def count_by_type(self, att_type: AttachmentType) -> int:
        return sum(1 for a in self.attachments if a.attach_type == att_type)

    @property
    def total_count(self) -> int:
        return len(self.attachments)

    def to_dict(self) -> List[Dict]:
        return [a.to_dict() for a in self.attachments]

    @classmethod
    def from_dict(cls, data: List[Dict]) -> "AttachmentManager":
        mgr = cls()
        for d in data:
            att = Attachment.from_dict(d)
            if os.path.exists(att.file_path):
                mgr.attachments.append(att)
        return mgr


# ── GUI 附件对话框 ──

class AttachmentDialog(QDialog):
    """附件管理对话框"""

    def __init__(self, attachment_mgr: AttachmentManager, parent=None):
        super().__init__(parent)
        self.mgr = attachment_mgr
        self._init_ui()
        self._refresh()

    def _init_ui(self):
        self.setWindowTitle("📎 附件管理")
        self.setMinimumSize(600, 400)
        self.resize(700, 500)

        self.setStyleSheet("""
            QDialog {
                background-color: #0d0d0d;
                color: #d4d4d4;
            }
            QLabel {
                color: #c89b3c;
            }
            QListWidget {
                background-color: #111111;
                color: #d4d4d4;
                border: 1px solid #2a2a2a;
                border-radius: 4px;
                font-size: 12px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #1e1e1e;
            }
            QListWidget::item:selected {
                background-color: #c89b3c;
                color: #0d0d0d;
            }
            QPushButton {
                background-color: #c89b3c;
                color: #0d0d0d;
                border: none;
                border-radius: 4px;
                padding: 6px 16px;
                font-weight: bold;
                min-height: 28px;
            }
            QPushButton:hover {
                background-color: #dbb052;
            }
            QPushButton.danger {
                background-color: #c0392b;
            }
            QPushButton.danger:hover {
                background-color: #e74c3c;
            }
            QPushButton.secondary {
                background-color: #2a2a2a;
                color: #d4d4d4;
            }
            QPushButton.secondary:hover {
                background-color: #3a3a3a;
            }
            QTextEdit {
                background-color: #111111;
                color: #d4d4d4;
                border: 1px solid #2a2a2a;
                border-radius: 4px;
                padding: 6px;
                font-size: 12px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)

        title = QLabel("📎 文件附件管理")
        title_font = QFont("Microsoft YaHei", 13, QFont.Weight.Bold)
        title.setFont(title_font)
        layout.addWidget(title)

        info = QLabel("将文件附加到讨论中，AI专家将能读取附件内容作为讨论上下文")
        info.setStyleSheet("color: #888888; font-size: 11px;")
        info.setWordWrap(True)
        layout.addWidget(info)

        # 按钮行
        btn_row = QHBoxLayout()
        self.btn_add_file = QPushButton("📄 添加文件")
        self.btn_add_file.clicked.connect(self._add_file)
        btn_row.addWidget(self.btn_add_file)

        self.btn_add_image = QPushButton("🖼️ 添加图片")
        self.btn_add_image.clicked.connect(self._add_image)
        btn_row.addWidget(self.btn_add_image)

        self.btn_remove = QPushButton("🗑️ 移除选中", clicked=self._remove_selected)
        self.btn_remove.setStyleSheet("background-color: #c0392b; color: white;")
        btn_row.addWidget(self.btn_remove)

        self.btn_clear = QPushButton("清空全部", clicked=self._clear_all)
        self.btn_clear.setStyleSheet("background-color: #2a2a2a; color: #d4d4d4;")
        btn_row.addWidget(self.btn_clear)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        # 附件列表
        self.att_list = QListWidget()
        self.att_list.currentRowChanged.connect(self._on_selected)
        layout.addWidget(self.att_list, 1)

        # 选中附件的描述/预览
        self.preview_label = QLabel("选中附件后可编辑描述")
        self.preview_label.setStyleSheet("color: #555555; font-size: 11px; padding: 4px;")
        self.preview_label.setWordWrap(True)
        layout.addWidget(self.preview_label)

        desc_row = QHBoxLayout()
        desc_row.addWidget(QLabel("描述:"))
        self.desc_input = QTextEdit()
        self.desc_input.setMaximumHeight(50)
        self.desc_input.setPlaceholderText("可选：为附件添加描述说明")
        desc_row.addWidget(self.desc_input, 1)
        self.btn_update_desc = QPushButton("更新描述", clicked=self._update_desc)
        self.btn_update_desc.setStyleSheet("background-color: #2a2a2a; color: #d4d4d4;")
        desc_row.addWidget(self.btn_update_desc)
        layout.addLayout(desc_row)

        # 关闭
        close_row = QHBoxLayout()
        close_row.addStretch()
        self.btn_close = QPushButton("关闭", clicked=self.close)
        close_row.addWidget(self.btn_close)
        layout.addLayout(close_row)

    def _refresh(self):
        self.att_list.clear()
        for i, att in enumerate(self.mgr.attachments):
            icon = "🖼️" if att.attach_type == AttachmentType.IMAGE else "📄"
            desc = att.description[:40] if att.description else att.file_name[:40]
            item = QListWidgetItem(f"{icon} [{att.type_name}] {desc}")
            item.setToolTip(att.file_path)
            self.att_list.addItem(item)

        self.preview_label.setText(
            f"共 {self.mgr.total_count} 个附件 "
            f"(图片:{self.mgr.count_by_type(AttachmentType.IMAGE)} "
            f"文本:{self.mgr.count_by_type(AttachmentType.TEXT)}"
            f" 其他:{self.mgr.count_by_type(AttachmentType.OTHER)})"
        )

    def _add_file(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择文件", "",
            "所有文件 (*.*);;文本文件 (*.txt *.md *.py *.json);;数据文件 (*.csv *.json *.xml)",
        )
        for path in paths:
            self.mgr.add(path)
        self._refresh()

    def _add_image(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择图片", "",
            "图片文件 (*.png *.jpg *.jpeg *.gif *.bmp *.webp)",
        )
        for path in paths:
            self.mgr.add(path)
        self._refresh()

    def _remove_selected(self):
        row = self.att_list.currentRow()
        if row >= 0:
            self.mgr.remove(row)
            self._refresh()

    def _clear_all(self):
        if self.mgr.total_count == 0:
            return
        reply = QMessageBox.question(
            self, "确认清空", "确定要移除所有附件吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.mgr.clear()
            self._refresh()

    def _on_selected(self, row: int):
        if 0 <= row < len(self.mgr.attachments):
            att = self.mgr.attachments[row]
            self.desc_input.setText(att.description)
            preview = f"📄 {att.file_name}\n📁 {att.file_path}\n📋 类型: {att.type_name}\n🕐 {att.attached_at}"
            self.preview_label.setText(preview)
        else:
            self.desc_input.clear()
            self.preview_label.setText("选中附件后可编辑描述")

    def _update_desc(self):
        row = self.att_list.currentRow()
        if 0 <= row < len(self.mgr.attachments):
            desc = self.desc_input.toPlainText().strip()
            if desc:
                self.mgr.attachments[row].description = desc
            self._refresh()


# ── 图片预览对话框 ──

class ImagePreviewDialog(QDialog):
    """图片预览对话框"""

    def __init__(self, pixmap: QPixmap, title: str = "图片预览", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(400, 300)
        self.resize(800, 600)

        self.setStyleSheet("""
            QDialog {
                background-color: #0d0d0d;
                color: #d4d4d4;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background-color: #0d0d0d; border: none;")

        label = QLabel()
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scroll.setWidget(label)

        layout.addWidget(scroll, 1)

        close_row = QHBoxLayout()
        close_row.addStretch()
        btn_close = QPushButton("关闭", clicked=self.close)
        btn_close.setStyleSheet("""
            QPushButton {
                background-color: #c89b3c; color: #0d0d0d;
                border: none; border-radius: 4px;
                padding: 6px 20px; font-weight: bold;
            }
        """)
        close_row.addWidget(btn_close)
        layout.addLayout(close_row)


# ── TTS 设置对话框 ──

class TTSDialog(QDialog):
    """语音输出设置对话框（支持在线TTS配置）"""

    def __init__(self, tts: TextToSpeech, parent=None):
        super().__init__(parent)
        self.tts = tts
        self._init_ui()

    def _init_ui(self):
        self.setWindowTitle("🔊 语音输出设置")
        self.setMinimumSize(500, 420)
        self.resize(540, 460)

        self.setStyleSheet("""
            QDialog {
                background-color: #0d0d0d;
                color: #d4d4d4;
            }
            QLabel { color: #c89b3c; }
            QGroupBox {
                color: #c89b3c; font-weight: bold;
                border: 1px solid #2a2a2a; border-radius: 4px;
                margin-top: 12px; padding-top: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 10px; padding: 0 6px;
            }
            QPushButton {
                background-color: #c89b3c; color: #0d0d0d;
                border: none; border-radius: 4px;
                padding: 6px 16px; font-weight: bold; min-height: 28px;
            }
            QPushButton:hover { background-color: #dbb052; }
            QPushButton:disabled { background-color: #333; color: #666; }
            QCheckBox { color: #d4d4d4; }
            QLineEdit, QTextEdit {
                background-color: #111111; color: #d4d4d4;
                border: 1px solid #2a2a2a; border-radius: 4px;
                padding: 4px 8px; font-size: 12px;
            }
            QTabWidget::pane {
                background-color: #111111; border: 1px solid #2a2a2a;
                border-radius: 4px;
            }
            QTabBar::tab {
                background-color: #1a1a1a; color: #888;
                padding: 6px 16px; border: 1px solid #2a2a2a;
                border-bottom: none; border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #111111; color: #c89b3c;
            }
            QSlider::groove:horizontal {
                height: 6px; background: #2a2a2a; border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #c89b3c; width: 14px; height: 14px;
                margin: -4px 0; border-radius: 7px;
            }
            QSlider::sub-page:horizontal {
                background: #c89b3c; border-radius: 3px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)

        title = QLabel("🔊 语音输出设置")
        title_font = QFont("Microsoft YaHei", 13, QFont.Weight.Bold)
        title.setFont(title_font)
        layout.addWidget(title)

        info = QLabel(f"当前引擎: {self.tts.provider_name}")
        info.setStyleSheet("color: #888888; font-size: 11px;")
        layout.addWidget(info)

        # 标签页
        tabs = QTabWidget()
        layout.addWidget(tabs, 1)

        # ── 标签页 1: 基础设置 ──
        basic_tab = QWidget()
        basic_layout = QVBoxLayout(basic_tab)
        basic_layout.setSpacing(8)

        if not self.tts.available and self.tts._provider != TTSProvider.ONLINE_OPENAI:
            warn = QLabel("⚠️ 当前系统无可用语音引擎。\n建议安装 pyttsx3: pip install pyttsx3\n或在「在线 TTS」标签页中配置在线语音。")
            warn.setStyleSheet("color: #e67e22; font-size: 11px;")
            warn.setWordWrap(True)
            basic_layout.addWidget(warn)

        # 启用开关
        self.enable_check = QCheckBox("启用语音输出（TTS）")
        self.enable_check.setChecked(self.tts.enabled)
        self.enable_check.toggled.connect(self._on_toggle)
        basic_layout.addWidget(self.enable_check)

        # 语速
        speed_group = QGroupBox("语速")
        speed_layout = QVBoxLayout(speed_group)
        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("慢"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(50, 300)
        self.speed_slider.setValue(self.tts._rate)
        self.speed_slider.valueChanged.connect(self._on_speed_changed)
        speed_row.addWidget(self.speed_slider, 1)
        speed_row.addWidget(QLabel("快"))
        self.speed_label = QLabel(f"{self.tts._rate}")
        self.speed_label.setStyleSheet("color: #888888; min-width: 30px;")
        speed_row.addWidget(self.speed_label)
        speed_layout.addLayout(speed_row)
        basic_layout.addWidget(speed_group)

        # 音量
        vol_group = QGroupBox("音量")
        vol_layout = QVBoxLayout(vol_group)
        vol_row = QHBoxLayout()
        vol_row.addWidget(QLabel("低"))
        self.vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.vol_slider.setRange(0, 100)
        self.vol_slider.setValue(int(self.tts._volume * 100))
        self.vol_slider.valueChanged.connect(self._on_vol_changed)
        vol_row.addWidget(self.vol_slider, 1)
        vol_row.addWidget(QLabel("高"))
        self.vol_label = QLabel(f"{int(self.tts._volume * 100)}%")
        self.vol_label.setStyleSheet("color: #888888; min-width: 30px;")
        vol_row.addWidget(self.vol_label)
        vol_layout.addLayout(vol_row)
        basic_layout.addWidget(vol_group)

        # 测试按钮
        test_row = QHBoxLayout()
        self.btn_test = QPushButton("🔊 测试语音")
        self.btn_test.clicked.connect(self._test_voice)
        test_row.addWidget(self.btn_test)
        test_row.addStretch()
        basic_layout.addLayout(test_row)

        basic_layout.addStretch()
        tabs.addTab(basic_tab, "基础设置")

        # ── 标签页 2: 在线 TTS ──
        online_tab = QWidget()
        online_layout = QVBoxLayout(online_tab)
        online_layout.setSpacing(8)

        online_info = QLabel("在线 TTS 通过 xh.v1api.cc 代理调用 OpenAI 兼容的语音合成接口")
        online_info.setStyleSheet("color: #888888; font-size: 11px;")
        online_info.setWordWrap(True)
        online_layout.addWidget(online_info)

        # API Key
        online_layout.addWidget(QLabel("API Key:"))
        self.online_key_input = QTextEdit()
        self.online_key_input.setMaximumHeight(36)
        self.online_key_input.setPlainText(self.tts._online_api_key)
        self.online_key_input.setPlaceholderText("输入在线 TTS 的 API Key")
        online_layout.addWidget(self.online_key_input)

        # Base URL
        online_layout.addWidget(QLabel("Base URL:"))
        url_row = QHBoxLayout()
        self.online_url_input = QTextEdit()
        self.online_url_input.setMaximumHeight(36)
        self.online_url_input.setPlainText(self.tts._online_base_url)
        self.online_url_input.setPlaceholderText("https://xh.v1api.cc")
        url_row.addWidget(self.online_url_input, 1)
        online_layout.addLayout(url_row)

        # Voice
        voice_row = QHBoxLayout()
        voice_row.addWidget(QLabel("音色:"))
        from PySide6.QtWidgets import QComboBox
        self.voice_combo = QComboBox()
        self.voice_combo.addItems(["alloy", "echo", "fable", "onyx", "nova", "shimmer"])
        current_voice = self.tts._online_voice
        idx = self.voice_combo.findText(current_voice)
        if idx >= 0:
            self.voice_combo.setCurrentIndex(idx)
        voice_row.addWidget(self.voice_combo, 1)
        online_layout.addLayout(voice_row)

        # 模型
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("模型:"))
        self.model_input = QTextEdit()
        self.model_input.setMaximumHeight(36)
        self.model_input.setPlainText(self.tts._online_model)
        self.model_input.setPlaceholderText("tts-1, tts-1-hd")
        model_row.addWidget(self.model_input, 1)
        online_layout.addLayout(model_row)

        # 应用按钮
        apply_row = QHBoxLayout()
        self.btn_apply_online = QPushButton("💾 应用在线 TTS 设置")
        self.btn_apply_online.clicked.connect(self._apply_online_settings)
        apply_row.addWidget(self.btn_apply_online)
        apply_row.addStretch()
        online_layout.addLayout(apply_row)

        online_layout.addStretch()
        tabs.addTab(online_tab, "在线 TTS")

        # 关闭按钮
        close_row = QHBoxLayout()
        close_row.addStretch()
        self.btn_close = QPushButton("关闭", clicked=self.close)
        close_row.addWidget(self.btn_close)
        layout.addLayout(close_row)

    def _on_toggle(self, enabled: bool):
        self.tts.enabled = enabled

    def _on_speed_changed(self, val: int):
        self.tts.set_rate(val)
        self.speed_label.setText(str(val))

    def _on_vol_changed(self, val: int):
        self.tts.set_volume(val / 100)
        self.vol_label.setText(f"{val}%")

    def _apply_online_settings(self):
        """应用在线 TTS 配置"""
        api_key = self.online_key_input.toPlainText().strip()
        base_url = self.online_url_input.toPlainText().strip()
        voice = self.voice_combo.currentText()
        model = self.model_input.toPlainText().strip() or "tts-1"
        if not api_key:
            QMessageBox.warning(self, "提示", "请输入 API Key")
            return
        if not base_url:
            base_url = _get_b64_prompt("TTS_BASE_URL")
        self.tts.configure_online(api_key=api_key, base_url=base_url, voice=voice, model=model)
        QMessageBox.information(self, "完成", f"在线 TTS 已配置\n引擎: {self.tts.provider_name}")

    def _test_voice(self):
        self.tts.speak("语音输出测试。如果听到声音，说明设置正确。")


# ── 全局实例 ──

_attachment_mgr = AttachmentManager()


def get_attachment_manager() -> AttachmentManager:
    return _attachment_mgr
