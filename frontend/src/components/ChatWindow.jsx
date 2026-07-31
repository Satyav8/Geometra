import { useEffect, useRef, useState } from "react";
import { v4 as uuidv4 } from "uuid";
import { sendMessage } from "../api/chatApi.js";
import MessageBubble from "./MessageBubble.jsx";
import TypingIndicator from "./TypingIndicator.jsx";

export default function ChatWindow() {
  const [sessionId] = useState(() => uuidv4());
  const [turnNumber, setTurnNumber] = useState(1);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping]);

  const handleSend = async () => {
    const query = input.trim();
    if (!query || isTyping) return;

    setMessages((prev) => [...prev, { role: "user", text: query }]);
    setInput("");
    setIsTyping(true);

    try {
      const data = await sendMessage(sessionId, query, turnNumber);
      setMessages((prev) => [
        ...prev,
        {
          role: "bot",
          text: data.response,
          sources: data.sources,
          confidence_level: data.confidence_level,
          is_unknown_question: data.is_unknown_question,
          support_email: data.support_email,
        },
      ]);
      setTurnNumber((t) => t + 1);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "bot",
          text: "Something went wrong reaching the Geometra support backend. Please try again.",
          sources: [],
          confidence_level: "unknown",
          is_unknown_question: false,
        },
      ]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="chat-window">
      <div className="chat-header">
        <div className="chat-header-title">S.A.M</div>
        <div className="chat-header-subtitle">Simple Answering Machine · Geometra Support</div>
      </div>
      <div className="message-list">
        {messages.map((m, i) => (
          <MessageBubble key={i} message={m} />
        ))}
        {isTyping && <TypingIndicator />}
        <div ref={bottomRef} />
      </div>
      <div className="input-bar">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about Geometra..."
          rows={1}
        />
        <button onClick={handleSend} disabled={isTyping || !input.trim()}>
          Send
        </button>
      </div>
    </div>
  );
}
