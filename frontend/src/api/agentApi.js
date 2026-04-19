/**
 * agentApi.js — HTTP client for the Knowledge Graph Agent API.
 *
 * Usage:
 *   import { sendMessage, checkHealth } from './api/agentApi';
 *
 *   const { reply, intent, session_id } = await sendMessage("Sama works at Google", sessionId);
 */

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

/**
 * Send a chat message to the agent.
 *
 * @param {string} message    - User's message text.
 * @param {string} sessionId  - Conversation session ID. Pass the same ID to keep context.
 * @returns {Promise<{ reply: string, session_id: string, intent: string, fallback_used: string }>}
 */
export async function sendMessage(message, sessionId) {
  const response = await fetch(`${BASE_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, session_id: sessionId }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

/**
 * Check API and Neo4j health.
 *
 * @returns {Promise<{ status: string, neo4j: string }>}
 */
export async function checkHealth() {
  const response = await fetch(`${BASE_URL}/health`);
  if (!response.ok) throw new Error(`Health check failed: HTTP ${response.status}`);
  return response.json();
}

/**
 * Signal the server to clear the session (client should also generate a new sessionId).
 *
 * @param {string} sessionId
 */
export async function clearSession(sessionId) {
  await fetch(`${BASE_URL}/session/${sessionId}`, { method: "DELETE" });
}