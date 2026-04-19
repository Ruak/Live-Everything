import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { Components } from 'react-markdown';

const components: Components = {
  p: ({ children }) => (
    <p className="mb-2 last:mb-0 leading-relaxed text-white/90">{children}</p>
  ),
  strong: ({ children }) => (
    <strong className="font-semibold text-white">{children}</strong>
  ),
  em: ({ children }) => (
    <em className="italic text-cyan-100/90">{children}</em>
  ),
  ul: ({ children }) => (
    <ul className="mb-2 list-disc pl-4 space-y-1.5 text-white/[0.88] [&_ul]:mt-1.5">{children}</ul>
  ),
  ol: ({ children }) => (
    <ol className="mb-2 list-decimal pl-4 space-y-1.5 text-white/[0.88] [&_ol]:mt-1.5">{children}</ol>
  ),
  li: ({ children }) => (
    <li className="leading-relaxed text-white/[0.88] marker:text-cyan-400/70 [&>p]:text-white/[0.88]">
      {children}
    </li>
  ),
  h1: ({ children }) => (
    <h4 className="text-white font-semibold text-[15px] mb-1.5 mt-2 first:mt-0">
      {children}
    </h4>
  ),
  h2: ({ children }) => (
    <h4 className="text-white font-semibold text-[15px] mb-1.5 mt-2 first:mt-0">
      {children}
    </h4>
  ),
  h3: ({ children }) => (
    <h4 className="text-white font-semibold text-[14px] mb-1 mt-2 first:mt-0">
      {children}
    </h4>
  ),
  a: ({ href, children }) => (
    <a
      href={href}
      className="text-cyan-300 underline underline-offset-2 hover:text-cyan-200"
      target="_blank"
      rel="noopener noreferrer"
    >
      {children}
    </a>
  ),
  blockquote: ({ children }) => (
    <blockquote className="border-l-2 border-cyan-400/40 pl-2.5 my-2 text-white/75 text-[13px]">
      {children}
    </blockquote>
  ),
  code: ({ className, children }) => {
    const inline = !className;
    if (inline) {
      return (
        <code className="rounded bg-white/10 px-1 py-0.5 text-[12.5px] text-cyan-100">
          {children}
        </code>
      );
    }
    return (
      <code className={className}>{children}</code>
    );
  },
  pre: ({ children }) => (
    <pre className="mb-2 overflow-x-auto rounded-lg bg-black/35 p-2.5 text-[12px] text-white/90">
      {children}
    </pre>
  ),
  hr: () => <hr className="my-3 border-cyan-400/20" />,
};

type Props = {
  text: string;
  className?: string;
};

/** 渲染模型返回的 Markdown（加粗、列表、链接等），用于 AR 侧栏问答区 */
export function AnswerMarkdown({ text, className = '' }: Props) {
  return (
    <div
      className={`answer-markdown text-white/[0.88] selection:bg-cyan-500/25 ${className}`}
    >
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {text}
      </ReactMarkdown>
    </div>
  );
}
