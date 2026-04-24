import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from admin.config import EMAIL_FROM, EMAIL_PASSWORD, SMTP_HOST, SMTP_PORT, FRONTEND_URL

logger = logging.getLogger(__name__)


def send_verification_email(to_email: str, token: str) -> bool:
    subject = "Verify Your Email - Admin Dashboard"
    verify_url = f"{FRONTEND_URL}/verify-email/{token}"
    html = f"""
    <h2>Email Verification</h2>
    <p>Click the link below to verify your email address:</p>
    <a href="{verify_url}">{verify_url}</a>
    <p>This link will expire in 24 hours.</p>
    """
    return _send_email(to_email, subject, html)


def send_password_reset_email(to_email: str, token: str) -> bool:
    subject = "Password Reset - Admin Dashboard"
    reset_url = f"{FRONTEND_URL}/reset-password/{token}"
    html = f"""
    <h2>Password Reset</h2>
    <p>Click the link below to reset your password:</p>
    <a href="{reset_url}">{reset_url}</a>
    <p>This link will expire in 1 hour.</p>
    """
    return _send_email(to_email, subject, html)


def _send_email(to_email: str, subject: str, html: str) -> bool:
    if not EMAIL_FROM or not EMAIL_PASSWORD:
        logger.warning("Email not configured, skipping send to %s", to_email)
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = to_email
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_FROM, EMAIL_PASSWORD)
            server.sendmail(EMAIL_FROM, to_email, msg.as_string())
        return True
    except Exception as e:
        logger.error("Failed to send email to %s: %s", to_email, e)
        return False
