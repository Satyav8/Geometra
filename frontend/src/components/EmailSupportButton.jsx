export default function EmailSupportButton({ email }) {
  if (!email) return null;

  const gmailComposeUrl = `https://mail.google.com/mail/?view=cm&fs=1&to=${encodeURIComponent(email)}`;

  return (
    <a
      className="email-support-button"
      href={gmailComposeUrl}
      target="_blank"
      rel="noopener noreferrer"
    >
      📧 Email Support Team
    </a>
  );
}
