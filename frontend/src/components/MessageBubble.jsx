import EmailSupportButton from "./EmailSupportButton.jsx";
import SourceCitation from "./SourceCitation.jsx";

export default function MessageBubble({ message }) {
  const isUser = message.role === "user";

  if (isUser) {
    return (
      <div className="message-row user-row">
        <div className="bubble user-bubble">{message.text}</div>
      </div>
    );
  }

  return (
    <div className="message-row bot-row">
      <div className="bubble bot-bubble">
        <div className="bot-text">{message.text}</div>
        {message.is_unknown_question && (
          <div className="unknown-notice">
            This question has been flagged for our support team.
          </div>
        )}
        <EmailSupportButton email={message.support_email} />
        <SourceCitation sources={message.sources} />
      </div>
    </div>
  );
}
