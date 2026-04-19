import { useCallback, useRef } from 'react';
import { Mic, MicOff, Loader2 } from 'lucide-react';
import { useAppStore } from '../store/useAppStore';
import { useVoice } from '../hooks/useVoice';
import { generateAnswer } from '../services/qaService';
import { askAgentAudio } from '../services/agentService';
import type { Product } from '../types/product';

export function VoiceButton() {
  const {
    state,
    transition,
    currentProduct,
    currentAgentId,
    backendReady,
    setVoiceText,
    setCurrentAnswer,
    addQA,
  } = useAppStore();

  const { isRecording, startRecording, stopRecording } = useVoice();

  /** 把手里最新的依赖放进 ref，给注册在 useVoice 里的 autoFinish 回调用。 */
  const ctxRef = useRef({
    currentProduct,
    currentAgentId,
    backendReady,
  });
  ctxRef.current = { currentProduct, currentAgentId, backendReady };

  /** 把"blob → 后端 → 展示"的流程抽出来；手动结束与自动超时都走它。 */
  const processBlob = useCallback(
    async (blob: Blob, product: Product | null, agentId: string | null, backendUp: boolean) => {
      if (!product) {
        // 物体已消失，丢掉这段录音
        transition('InfoPanelOpen');
        return;
      }

      transition('VoiceProcessing');

      if (blob.size < 1024) {
        console.warn('[voice] 录到的音频太小，跳过');
        transition('InfoPanelOpen');
        return;
      }

      try {
        let transcription = '';
        let answer = '';

        if (agentId && backendUp) {
          const resp = await askAgentAudio(agentId, blob);
          transcription = (resp.transcription || '').trim();
          answer = (resp.answer || '').trim();
        }

        if (!transcription) {
          console.warn('[voice] 未获得转写结果', { agentId, backendUp });
          const fallback = !agentId
            ? '语音服务未连接。请确认 Agent 后端已启动（默认 http://localhost:8000）。'
            : '未能识别到语音内容，请靠近麦克风再试一次。';
          setCurrentAnswer(fallback);
          addQA('（未识别到语音）', fallback);
          transition('AnswerDisplayed');
          return;
        }

        if (!answer) {
          answer = generateAnswer(product, transcription);
        }

        setVoiceText(transcription);
        setCurrentAnswer(answer);
        addQA(transcription, answer);
        transition('AnswerDisplayed');
      } catch (err) {
        console.error('[voice] 后端语音问答失败', err);
        const msg = '语音服务暂不可用，请稍后重试。';
        setCurrentAnswer(msg);
        addQA('（请求失败）', msg);
        transition('AnswerDisplayed');
      }
    },
    [transition, setVoiceText, setCurrentAnswer, addQA]
  );

  const handleToggle = useCallback(async () => {
    // 已在录音中 → 手动结束
    if (state === 'VoiceRecording' || isRecording) {
      const blob = await stopRecording();
      if (!blob) {
        // 上限已到 autoFinish 代劳；或根本没 recorder：回退状态让用户能再次录
        if (state === 'VoiceRecording') transition('InfoPanelOpen');
        return;
      }
      await processBlob(blob, currentProduct, currentAgentId, backendReady);
      return;
    }

    // 未在录音 → 启动，顺带注册 autoFinish 兜底
    if (state === 'InfoPanelOpen' || state === 'AnswerDisplayed') {
      const started = await startRecording(async (blob) => {
        const ctx = ctxRef.current;
        await processBlob(blob, ctx.currentProduct, ctx.currentAgentId, ctx.backendReady);
      });
      if (started) {
        transition('VoiceRecording');
      } else {
        setCurrentAnswer(
          '麦克风启动失败：请在浏览器允许麦克风权限（地址栏左侧的锁图标 → 网站设置 → 麦克风：允许），并确认没有其他程序占用。'
        );
        transition('AnswerDisplayed');
      }
    }
  }, [
    state,
    isRecording,
    currentProduct,
    currentAgentId,
    backendReady,
    startRecording,
    stopRecording,
    processBlob,
    transition,
    setCurrentAnswer,
  ]);

  const isProcessing = state === 'VoiceProcessing';
  const canRecord = state === 'InfoPanelOpen' || state === 'AnswerDisplayed';
  const disabled = isProcessing || (!canRecord && !isRecording);

  const hint = !backendReady
    ? '本地问答'
    : currentAgentId
      ? '云端问答'
      : '初始化中…';

  return (
    <div className="flex items-center gap-3">
      <button
        onClick={handleToggle}
        disabled={disabled}
        title={hint}
        className={`
          flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl
          font-medium text-sm transition-all duration-200
          ${
            isRecording
              ? 'bg-red-500/90 text-white voice-recording'
              : isProcessing
                ? 'bg-white/10 text-white/40 cursor-wait'
                : canRecord
                  ? 'bg-ar-primary/90 text-black hover:bg-ar-primary'
                  : 'bg-white/10 text-white/30 cursor-not-allowed'
          }
        `}
      >
        {isRecording ? (
          <>
            <MicOff size={16} />
            点击结束录音
          </>
        ) : isProcessing ? (
          <>
            <Loader2 size={16} className="animate-spin" />
            识别中…
          </>
        ) : (
          <>
            <Mic size={16} />
            语音提问
          </>
        )}
      </button>

      {isRecording && (
        <div className="flex items-center gap-1">
          {[...Array(4)].map((_, i) => (
            <div
              key={i}
              className="w-1 bg-red-400 rounded-full animate-pulse"
              style={{
                height: `${12 + Math.random() * 12}px`,
                animationDelay: `${i * 0.15}s`,
              }}
            />
          ))}
        </div>
      )}

      <span className="text-[11px] text-white/40 whitespace-nowrap">{hint}</span>
    </div>
  );
}
