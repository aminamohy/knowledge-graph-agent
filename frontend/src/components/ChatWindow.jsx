/**
 * ChatWindow.jsx — Main chat interface.
 *
 * Manages:
 *  - Message list state
 *  - Session ID (persisted in localStorage)
 *  - API calls via agentApi
 *  - Neo4j health badge
 */

import React, { useState, useEffect, useRef, useCallback } from "react";
import MessageBubble from "./MessageBubble";
import InputBar from "./InputBar";
import { sendMessage, checkHealth, clearSession } from "../api/agentApi";

function generateSessionId() {
  return `session-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

export default function ChatWindow() {
  const [messages, setMessages]     = useState([]);
  const [loading, setLoading]       = useState(false);
  const [sessionId, setSessionId]   = useState(() => {
    return localStorage.getItem("kg_session_id") || generateSessionId();
  });
  const [health, setHealth]         = useState(null); // null | "connected" | "unreachable"
  const bottomRef                   = useRef(null);

  // Persist session ID
  useEffect(() => {
    localStorage.setItem("kg_session_id", sessionId);
  }, [sessionId]);

  // Health check on mount
  useEffect(() => {
    checkHealth()
      .then(data => setHealth(data.neo4j))
      .catch(() => setHealth("unreachable"));
  }, []);

  // Auto-scroll to latest message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const handleSend = useCallback(async (text) => {
    const userMsg = {
      id:        Date.now(),
      role:      "user",
      content:   text,
      timestamp: new Date(),
    };
    setMessages(prev => [...prev, userMsg]);
    setLoading(true);

    try {
      const data = await sendMessage(text, sessionId);
      const agentMsg = {
        id:        Date.now() + 1,
        role:      "assistant",
        content:   data.reply,
        intent:    data.intent,
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, agentMsg]);
    } catch (err) {
      const errMsg = {
        id:        Date.now() + 1,
        role:      "assistant",
        content:   `⚠️ Error: ${err.message}. Please try again.`,
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errMsg]);
    } finally {
      setLoading(false);
    }
  }, [sessionId]);

  const handleClearSession = useCallback(async () => {
    await clearSession(sessionId).catch(() => {});
    const newId = generateSessionId();
    setSessionId(newId);
    setMessages([]);
  }, [sessionId]);

  const healthColor = health === "connected" ? "#22c55e" : health === "unreachable" ? "#ef4444" : "#6b7280";
  const healthLabel = health ?? "checking…";

  return (
    <div className="chat-window">
      {/* ── Header ── */}
      <div className="chat-header">
        <div className="chat-header__left">
          <div className="chat-header__icon">KG</div>
          <div>
            <h1 className="chat-header__title">Knowledge Graph Agent</h1>
            <p className="chat-header__subtitle">Neo4j · LangGraph · GPT-4o-mini</p>
          </div>
        </div>
        <div className="chat-header__right">
          <div className="health-badge" style={{ borderColor: healthColor + "55", color: healthColor }}>
            <span className="health-dot" style={{ background: healthColor }} />
            Neo4j {healthLabel}
          </div>
          <button className="clear-btn" onClick={handleClearSession} title="Start new session">
            ↺ New session
          </button>
        </div>
      </div>

      {/* ── Message list ── */}
      <div className="chat-messages">
        {messages.length === 0 && !loading && (
          <div className="chat-empty">
            <div className="chat-empty__icon">🕸️</div>
            <h2>Start building your knowledge graph</h2>
            <p>Tell me a fact to store, ask a question, or say hello.</p>
            <div className="chat-suggestions">
              {["Sama works at Google", "Who works at Google?", "Sama now works at Meta"].map(s => (
                <button key={s} className="suggestion-chip" onClick={() => handleSend(s)}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map(msg => (
          <MessageBubble key={msg.id} message={msg} />
        ))}

        {loading && (
          <div className="msg-row msg-row--agent">
            <div className="msg-avatar msg-avatar--agent">KG</div>
            <div className="msg-body">
              <div className="msg-bubble msg-bubble--agent msg-bubble--loading">
                <span /><span /><span />
              </div>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* ── Input ── */}
      <InputBar onSend={handleSend} disabled={loading} />
    </div>
  );
}