import { useCallback } from 'react';
import { X, Sparkles, Target, Gauge, Users, Play, Lightbulb } from 'lucide-react';
import { useAppStore } from '../store/useAppStore';
import { VoiceButton } from './VoiceButton';
import { AnswerMarkdown } from './AnswerMarkdown';
import { CONFIG } from '../config';

export function InfoPanel() {
  const { state, currentProduct, transition, qaHistory } = useAppStore();

  const isOpen =
    state === 'InfoPanelOpen' ||
    state === 'VoiceRecording' ||
    state === 'VoiceProcessing' ||
    state === 'AnswerDisplayed';

  const handleClose = useCallback(() => {
    transition('QRCodeVisible');
  }, [transition]);

  if (!isOpen || !currentProduct) return null;

  const coverSrc = currentProduct.cover_image
    ? `${CONFIG.knowledgeBasePath}${currentProduct.cover_image}`
    : null;

  const introDemo = currentProduct.guided_demo_script.find(
    (g) => g.step.includes('长按') || g.step.includes('展开')
  );

  return (
    <div
      className="absolute right-4 top-4 bottom-4 w-[400px] pointer-events-auto z-20"
      style={{ animation: 'arHudBadgeIn 320ms ease-out' }}
    >
      <div className="ar-glass rounded-2xl h-full flex flex-col overflow-hidden">
        {/* Header */}
        <div className="relative flex items-start justify-between p-4 pb-3 border-b border-cyan-400/15">
          <div className="flex-1 min-w-0 pr-3">
            <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.2em] text-cyan-300/75 mb-1">
              <Sparkles size={12} />
              <span>AR 商品讲解</span>
            </div>
            <h2 className="text-white text-lg font-bold truncate">
              {currentProduct.product_name}
            </h2>
            <p className="text-cyan-200/90 text-sm mt-1 leading-snug line-clamp-2">
              {currentProduct.one_line_hook ?? currentProduct.tagline}
            </p>
          </div>
          <button
            onClick={handleClose}
            className="p-1.5 rounded-lg hover:bg-white/10 transition-colors text-white/60 hover:text-white"
          >
            <X size={18} />
          </button>
        </div>

        {/* Scrollable content */}
        <div className="flex-1 overflow-y-auto info-scroll p-4 space-y-5">
          {/* Cover */}
          {coverSrc && (
            <img
              src={coverSrc}
              alt={currentProduct.product_name}
              className="w-full h-40 object-cover rounded-xl"
              onError={(e) => {
                (e.currentTarget as HTMLImageElement).style.display = 'none';
              }}
            />
          )}

          {/* 自我介绍（展开后的引导引语） */}
          {introDemo && (
            <div
              className="p-3 rounded-xl border border-cyan-400/25"
              style={{
                background:
                  'linear-gradient(135deg, rgba(0,255,225,0.08), rgba(124,58,237,0.1))',
              }}
            >
              <p className="text-white/90 text-sm leading-relaxed">
                <span className="mr-1 text-cyan-300">「</span>
                {introDemo.line}
                <span className="ml-1 text-cyan-300">」</span>
              </p>
            </div>
          )}

          {/* 简介 */}
          {currentProduct.self_intro_short && (
            <Section icon={<Sparkles size={13} />} title="一句话认识">
              <p className="text-white/80 text-sm leading-relaxed">
                {currentProduct.self_intro_short}
              </p>
            </Section>
          )}

          {/* 核心价值 chips */}
          {currentProduct.core_values.length > 0 && (
            <Section icon={<Target size={13} />} title="核心价值">
              <div className="flex flex-wrap gap-1.5">
                {currentProduct.core_values.map((v, i) => (
                  <span key={i} className="ar-chip">
                    {v}
                  </span>
                ))}
              </div>
            </Section>
          )}

          {/* 卖点 */}
          {currentProduct.selling_points.length > 0 && (
            <Section icon={<Lightbulb size={13} />} title="核心卖点">
              <ul className="space-y-2.5">
                {currentProduct.selling_points.map((pt, i) => (
                  <li
                    key={i}
                    className="pl-3 relative text-sm leading-relaxed"
                  >
                    <span
                      className="absolute left-0 top-1.5 h-3 w-[2px] rounded"
                      style={{
                        background:
                          'linear-gradient(180deg, #00ffe1, #7c3aed)',
                      }}
                    />
                    <p className="text-white/95 font-medium">{pt.title}</p>
                    <p className="text-white/60 text-[13px] mt-0.5">
                      {pt.detail}
                    </p>
                    {pt.scene_value && (
                      <p className="text-cyan-300/80 text-xs mt-1">
                        · {pt.scene_value}
                      </p>
                    )}
                  </li>
                ))}
              </ul>
            </Section>
          )}

          {/* 参数 */}
          {currentProduct.specs.length > 0 && (
            <Section icon={<Gauge size={13} />} title="基础参数">
              <div className="grid grid-cols-2 gap-x-4 gap-y-2">
                {currentProduct.specs.map((s, i) => (
                  <div key={i} className="text-[12.5px] leading-snug">
                    <div className="text-white/40">{s.name}</div>
                    <div className="text-white/90">{s.value}</div>
                  </div>
                ))}
              </div>
            </Section>
          )}

          {/* 适用人群 */}
          {currentProduct.audience.length > 0 && (
            <Section icon={<Users size={13} />} title="适用人群">
              <div className="flex flex-wrap gap-1.5">
                {currentProduct.audience.map((a, i) => (
                  <span key={i} className="ar-chip ar-chip--accent">
                    {a}
                  </span>
                ))}
              </div>
            </Section>
          )}

          {/* 应用场景 */}
          {currentProduct.use_cases.length > 0 && (
            <Section icon={<Play size={13} />} title="应用场景">
              <ul className="space-y-1.5">
                {currentProduct.use_cases.map((u, i) => (
                  <li
                    key={i}
                    className="text-white/75 text-[13px] flex items-start gap-2"
                  >
                    <span className="mt-[7px] w-1 h-1 rounded-full bg-cyan-300/70 flex-shrink-0" />
                    {u}
                  </li>
                ))}
              </ul>
            </Section>
          )}

          {/* 追问建议 */}
          {currentProduct.follow_up_questions &&
            currentProduct.follow_up_questions.length > 0 && (
              <Section title="你也可以继续问">
                <div className="space-y-1.5">
                  {currentProduct.follow_up_questions
                    .slice(0, 3)
                    .map((q, i) => (
                      <div
                        key={i}
                        className="text-[12.5px] text-cyan-200/85 bg-cyan-400/5 border border-cyan-400/15 rounded-lg px-2.5 py-1.5"
                      >
                        {q}
                      </div>
                    ))}
                </div>
              </Section>
            )}

          {/* Q&A History */}
          {qaHistory.length > 0 && (
            <Section title="问答记录">
              <div className="space-y-3">
                {qaHistory.map((qa, i) => {
                  const isLatest =
                    i === qaHistory.length - 1 && state === 'AnswerDisplayed';
                  return (
                    <div
                      key={i}
                      className={`space-y-1.5 rounded-xl p-2.5 -m-2 transition-colors ${
                        isLatest
                          ? 'bg-gradient-to-br from-cyan-400/10 to-violet-600/10 ring-1 ring-cyan-400/25'
                          : ''
                      }`}
                    >
                      <p className="text-cyan-300/90 text-xs">Q · {qa.q}</p>
                      <AnswerMarkdown text={qa.a} className="text-sm" />
                    </div>
                  );
                })}
              </div>
            </Section>
          )}

          {state === 'VoiceProcessing' && (
            <div className="flex items-center gap-2 text-white/60 text-sm">
              <div className="w-4 h-4 border-2 border-cyan-300 border-t-transparent rounded-full animate-spin" />
              正在处理语音…
            </div>
          )}
        </div>

        {/* Voice button area */}
        <div className="p-4 pt-3 border-t border-cyan-400/15">
          <VoiceButton />
        </div>
      </div>
    </div>
  );
}

function Section({
  icon,
  title,
  children,
}: {
  icon?: React.ReactNode;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <h3 className="ar-section-title flex items-center gap-1.5">
        {icon && <span className="text-cyan-300/70">{icon}</span>}
        {title}
      </h3>
      {children}
    </div>
  );
}
