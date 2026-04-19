/**
 * MessageBubble.jsx — Renders a single chat message bubble.
 *
 * Props:
 *   message: { role: "user" | "assistant", content: string, intent?: string, timestamp: Date }
 */

import React from "react";

const INTENT_COLORS = {
  add:          "#22c55e",   // green
  edit:         "#f59e0b",   // amber
  delete:       "#ef4444",   // red
  inquire:      "#3b82f6",   // blue
  chitchat:     "#8b5cf6",   // violet
  agent_info:   "#06b6d4",   // cyan
  out_of_scope: "#6b7280",   // gray
};

export default function MessageBubble({ message }) {
  const isUser = message.role === "user";
  const intentColor = INTENT_COLORS[message.intent] ?? "#6b7280";

  const time = message.timestamp
    ? new Date(message.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    : "";

  return (
    <div className={`msg-row ${isUser ? "msg-row--user" : "msg-row--agent"}`}>
      {/* Avatar */}
      <div className={`msg-avatar ${isUser ? "msg-avatar--user" : "msg-avatar--agent"}`}>
        {isUser ? "U" : "KG"}
      </div>

      <div className="msg-body">
        {/* Bubble */}
        <div className={`msg-bubble ${isUser ? "msg-bubble--user" : "msg-bubble--agent"}`}>
          <p className="msg-text">{message.content}</p>
        </div>

        {/* Meta row */}
        <div className={`msg-meta ${isUser ? "msg-meta--right" : "msg-meta--left"}`}>
          {!isUser && message.intent && (
            <span className="msg-intent" style={{ background: intentColor + "22", color: intentColor, borderColor: intentColor + "44" }}>
              {message.intent}
            </span>
          )}
          {time && <span className="msg-time">{time}</span>}
        </div>
      </div>
    </div>
  );
}