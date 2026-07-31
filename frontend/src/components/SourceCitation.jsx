import { useState } from "react";

export default function SourceCitation({ sources }) {
  const [expanded, setExpanded] = useState(false);

  if (!sources || sources.length === 0) return null;

  return (
    <div className="source-citation">
      <button className="source-toggle" onClick={() => setExpanded(!expanded)}>
        {expanded ? "▾" : "▸"} Sources used
      </button>
      {expanded && (
        <ul className="source-list">
          {sources.map((s) => (
            <li key={s.chunk_id} className="source-item">
              <span className="source-section">[{s.section}]</span>{" "}
              <span className="source-score">similarity: {s.similarity_score}</span>
              <div className="source-preview">{s.text.slice(0, 100)}{s.text.length > 100 ? "…" : ""}</div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
