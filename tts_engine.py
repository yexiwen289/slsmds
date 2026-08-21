"""
纯手工 TTS 引擎 —— 零模型、零 API、纯数学合成
============================================
原理：Klatt 共振峰合成（Klatt Formant Synthesis）
- 声门脉冲模型 → 级联共振峰滤波器 → 辐射特性
- 声母：白噪音 + 适当滤波
- 韵母：声门脉冲激励 → 3 个级联共振峰 → 声调控制
- 四声：基频轮廓控制

依赖：numpy, scipy, sounddevice, pypinyin（仅版本B需要）
"""

import numpy as np
from scipy import signal as sig


class ChineseTTS:
    """纯手工 TTS 引擎（版本A：拼音输入）"""

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

        # 声门脉冲参数
        self.f0_base = 120          # 基频（男声 ~ 120 Hz）
        self.open_quotient = 0.6    # 开商（open phase 比例）

        # ── 声母表 ──
        self.initials = {
            'b': 180, 'p': 170, 'm': 200, 'f': 190,
            'd': 185, 't': 175, 'n': 210, 'l': 215,
            'g': 160, 'k': 155, 'h': 170,
            'j': 220, 'q': 210, 'x': 230,
            'zh': 180, 'ch': 170, 'sh': 190, 'r': 200,
            'z': 185, 'c': 175, 's': 195,
            'y': 240, 'w': 230,
        }

        # ── 韵母表（F1, F2, F3, B1, B2, B3）──
        # 共振峰频率 + 带宽，单位 Hz
        self.finals = {
            'a':    (730, 1090, 2440,  80, 100, 120),
            'o':    (570,  840, 2410,  70,  90, 110),
            'e':    (530, 1840, 2480,  70, 110, 130),
            'i':    (290, 2360, 3010,  60, 130, 150),
            'u':    (320,  800, 2240,  60,  90, 110),
            'ü':    (350,  900, 2350,  60, 100, 120),
            'ai':   (650, 1200, 2400, 100, 120, 140),
            'ei':   (550, 1900, 2500,  90, 120, 140),
            'ao':   (700, 1150, 2450, 100, 110, 130),
            'ou':   (550,  900, 2300,  80, 100, 120),
            'an':   (700, 1200, 2500, 100, 120, 140),
            'en':   (550, 1800, 2500,  90, 120, 140),
            'ang':  (750, 1150, 2500, 100, 110, 130),
            'eng':  (550,  900, 2400,  80, 100, 120),
            'ong':  (500,  800, 2300,  80, 100, 120),
            'ia':   (400, 2000, 2800,  80, 130, 150),
            'ie':   (350, 2200, 2900,  70, 130, 150),
            'iao':  (400, 1900, 2800,  80, 120, 140),
            'iu':   (350, 1000, 2500,  70, 100, 120),
            'ian':  (400, 2000, 2800,  80, 130, 150),
            'in':   (350, 2200, 2900,  70, 130, 150),
            'iang': (450, 1900, 2800,  80, 120, 140),
            'ing':  (350, 2000, 2800,  70, 120, 140),
            'iong': (400,  900, 2400,  80, 100, 120),
            'ua':   (450, 1000, 2400,  80, 100, 120),
            'uo':   (500,  900, 2300,  80, 100, 120),
            'uai':  (400, 1100, 2400,  80, 100, 120),
            'ui':   (350,  900, 2300,  70, 100, 120),
            'uan':  (450, 1000, 2400,  80, 100, 120),
            'un':   (350,  900, 2300,  70, 100, 120),
            'uang': (450, 1000, 2400,  80, 100, 120),
            'üe':   (380, 1000, 2400,  70, 100, 120),
            'üan':  (400, 1000, 2450,  80, 100, 120),
            'ün':   (350,  950, 2400,  70, 100, 120),
        }

    # ═══════════════════════════════════════════
    #  拼音解析
    # ═══════════════════════════════════════════

    def parse_pinyin(self, pinyin):
        """解析拼音 → (声母, 韵母, 声调数字)"""
        pinyin = pinyin.strip().lower()
        if not pinyin:
            return ('', '', '1')

        tone = '1'
        if pinyin[-1].isdigit():
            tone = pinyin[-1]
            pinyin = pinyin[:-1]

        initial = ''
        final = pinyin

        if len(pinyin) >= 2 and pinyin[:2] in {'zh', 'ch', 'sh'}:
            initial = pinyin[:2]
            final = pinyin[2:]
        elif pinyin and pinyin[0] in self.initials:
            if pinyin[0] in ('y', 'w'):
                initial = pinyin[0]
                final = pinyin[1:]
                if not final:
                    final = pinyin[0]
            else:
                initial = pinyin[0]
                final = pinyin[1:]
                if not final:
                    final = pinyin[0]
        return initial, final, tone

    # ═══════════════════════════════════════════
    #  声调曲线
    # ═══════════════════════════════════════════

    def tone_curve(self, f0_base, tone, n_samples):
        """生成声调 F0 轮廓（相对值，乘以基频得到实际 F0）"""
        if tone == '1':       # 阴平 55
            return np.full(n_samples, 1.1)
        elif tone == '2':     # 阳平 35
            return np.linspace(0.9, 1.2, n_samples)
        elif tone == '3':     # 上声 214
            mid = int(n_samples * 0.4)
            c = np.empty(n_samples)
            c[:mid] = np.linspace(0.95, 0.65, mid)
            c[mid:] = np.linspace(0.65, 1.15, n_samples - mid)
            return c
        elif tone == '4':     # 去声 51
            return np.linspace(1.2, 0.7, n_samples)
        else:                 # 轻声
            return np.full(n_samples, 0.7) * 0.5

    # ═══════════════════════════════════════════
    #  声门脉冲模型（Rosenberg 型）
    # ═══════════════════════════════════════════

    def _glottal_pulse(self, period_samples):
        """
        生成单个 Rosenberg 型声门脉冲。
        形状：缓慢打开 → 快速关闭 → 关闭相位
        """
        n = int(period_samples)
        if n < 3:
            return np.ones(n)

        open_n = int(n * self.open_quotient)
        if open_n < 2:
            open_n = 2

        pulse = np.zeros(n)
        # 打开相位（前 1/3 的 open phase）
        open_rise = int(open_n * 0.4)
        if open_rise < 1:
            open_rise = 1
        # 关闭相位（后 2/3 的 open phase）
        open_fall = open_n - open_rise
        if open_fall < 1:
            open_fall = 1

        # 上升段：二次函数
        t_rise = np.arange(open_rise) / open_rise
        pulse[:open_rise] = 1.0 - (1.0 - t_rise) ** 2

        # 下降段：指数衰减
        t_fall = np.arange(open_fall) / open_fall
        pulse[open_rise:open_n] = np.exp(-3.0 * t_fall)

        # 关闭相位：零（保持声门关闭）
        return pulse

    def _glottal_excitation(self, f0_curve, n_samples):
        """
        生成完整的声门激励信号。
        用连续的声门脉冲序列模拟声带振动。
        """
        excitation = np.zeros(n_samples)
        t = 0.0
        idx = 0
        while idx < n_samples:
            f0 = f0_curve[idx]
            if f0 <= 0:
                idx += 1
                continue
            period = self.sample_rate / f0
            pulse = self._glottal_pulse(period)
            end = min(idx + len(pulse), n_samples)
            plen = end - idx
            excitation[idx:end] = pulse[:plen]
            idx = end
            t += period
        return excitation

    # ═══════════════════════════════════════════
    #  级联共振峰滤波器
    # ═══════════════════════════════════════════

    def _resonator(self, signal_in, freq, bw, sr):
        """
        二阶数字谐振器（单共振峰）。
        使用直接 II 型转置结构。
        """
        if freq <= 0 or freq >= sr / 2:
            return signal_in

        # 计算 IIR 系数
        theta = 2 * np.pi * freq / sr
        rho = np.exp(-np.pi * bw / sr)
        a1 = -2 * rho * np.cos(theta)
        a2 = rho * rho
        b0 = 1 - rho

        # 时域滤波
        n = len(signal_in)
        output = np.zeros(n)
        x1 = 0.0  # y[n-1]
        x2 = 0.0  # y[n-2]

        for i in range(n):
            y = b0 * signal_in[i] - a1 * x1 - a2 * x2
            output[i] = y
            x2 = x1
            x1 = y

        return output

    def _formant_filter(self, excitation, f1, f2, f3, b1, b2, b3):
        """
        级联共振峰滤波：F1 → F2 → F3
        每个共振峰是独立的二阶谐振器，串联处理。
        """
        sr = self.sample_rate
        out = self._resonator(excitation, f1, b1, sr)
        out = self._resonator(out, f2, b2, sr)
        out = self._resonator(out, f3, b3, sr)
        return out

    # ═══════════════════════════════════════════
    #  声母合成
    # ═══════════════════════════════════════════

    def synthesize_initial(self, initial, duration=0.06):
        """合成声母（白噪音 + 弱共振）"""
        if not initial or initial not in self.initials:
            return np.array([], dtype=np.float64)

        n = int(self.sample_rate * duration)
        noise = np.random.randn(n)

        # 用一个低通滤波器让噪音更像语音
        b, a = sig.butter(2, 4000, 'low', fs=self.sample_rate)
        noise = sig.filtfilt(b, a, noise)

        envelope = np.exp(-np.linspace(0, 6, n))
        return noise * envelope * 0.15

    # ═══════════════════════════════════════════
    #  韵母合成（核心）
    # ═══════════════════════════════════════════

    def synthesize_final(self, final, tone, duration=0.25):
        """
        合成韵母：
        1. 声调曲线 → 控制 F0
        2. 声门激励 → 脉冲序列
        3. 级联共振峰滤波 → F1/F2/F3
        4. 幅度包络
        """
        if not final or final not in self.finals:
            return np.array([], dtype=np.float64)

        n = int(self.sample_rate * duration)
        f1, f2, f3, b1, b2, b3 = self.finals[final]

        # 1. 声调曲线
        f0_rel = self.tone_curve(self.f0_base, tone, n)
        f0_curve = self.f0_base * f0_rel

        # 2. 声门激励
        excitation = self._glottal_excitation(f0_curve, n)

        # 3. 级联共振峰滤波
        filtered = self._formant_filter(excitation, f1, f2, f3, b1, b2, b3)

        # 4. 幅度包络
        envelope = np.hanning(n)
        return filtered * envelope * 0.6

    # ═══════════════════════════════════════════
    #  音节合成
    # ═══════════════════════════════════════════

    def synthesize_syllable(self, pinyin, duration=0.25):
        """合成一个音节：声母 + 韵母 + 声调"""
        initial, final, tone = self.parse_pinyin(pinyin)

        init_audio = self.synthesize_initial(initial, duration * 0.2)
        fin_audio = self.synthesize_final(final, tone, duration * 0.8)

        if len(init_audio) > 0:
            audio = np.concatenate([init_audio, fin_audio])
        else:
            audio = fin_audio

        if len(audio) > 0 and np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.6
        return audio

    # ═══════════════════════════════════════════
    #  文本合成
    # ═══════════════════════════════════════════

    def synthesize(self, text):
        """输入拼音串（空格分隔），合成完整音频"""
        syllables = text.strip().split()
        chunks = []
        for i, syl in enumerate(syllables):
            audio = self.synthesize_syllable(syl)
            if len(audio) > 0:
                chunks.append(audio)
            if i < len(syllables) - 1:
                chunks.append(np.zeros(int(self.sample_rate * 0.05)))
        return np.concatenate(chunks) if chunks else np.array([], dtype=np.float64)

    # ═══════════════════════════════════════════
    #  播放 / 保存
    # ═══════════════════════════════════════════

    def speak_pinyin(self, text):
        """播放拼音"""
        import sounddevice as sd
        audio = self.synthesize(text)
        if len(audio) > 0:
            sd.play(audio, self.sample_rate)
            sd.wait()
        return audio

    def save_wav(self, text, filepath):
        """保存为 WAV 文件"""
        import wave
        audio = self.synthesize(text)
        if len(audio) == 0:
            return False
        audio_int16 = (audio * 32767).astype(np.int16)
        with wave.open(filepath, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())
        return True


# ─────────────────────────────────────────────
#  版本 B：支持中文输入（需 pypinyin）
# ─────────────────────────────────────────────

class ChineseTTSWithChinese(ChineseTTS):
    """支持中文输入的 TTS 引擎"""

    def text_to_pinyin(self, text):
        """中文 → 拼音列表（带声调数字）"""
        from pypinyin import pinyin, Style
        result = pinyin(text, style=Style.TONE3, heteronym=False)
        return [item[0] for item in result]

    def synthesize_text(self, text):
        """输入中文，自动转拼音合成"""
        pylist = self.text_to_pinyin(text)
        pinyin_str = ' '.join(pylist)
        return self.synthesize(pinyin_str)

    def speak(self, text):
        """输入中文，自动朗读"""
        import sounddevice as sd
        audio = self.synthesize_text(text)
        if len(audio) > 0:
            sd.play(audio, self.sample_rate)
            sd.wait()
        return audio


# ════════════════════════════════════════════════════════════════
#  测试
# ════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    tts = ChineseTTSWithChinese()

    if len(sys.argv) > 1:
        text = ' '.join(sys.argv[1:])
    else:
        text = "ni3 hao3 wo3 shi4 yi1 ge5 zi4 wo3 yi4 shi2 xi4 tong3"

    print(f"朗读: {text}")
    tts.save_wav(text, "_tts_test.wav")
    print("已保存到 _tts_test.wav")
    tts.speak_pinyin(text)