import base64
import email
from dotenv import load_dotenv
from email.mime.text import MIMEText
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from typing import List, Dict

load_dotenv()

# ========= STEP 1: Gmail Setup =========

def get_service():
    creds = Credentials.from_authorized_user_file('token.json')
    return build('gmail', 'v1', credentials=creds)

# ---------- NEW: fetch a list instead of one -------------
def get_latest_emails(service, max_results: int = 3) -> List[Dict]:
    """Return a list of the N latest INBOX emails (subject, sender, id, threadId, body)."""
    result = (
        service.users()
        .messages()
        .list(userId="me", labelIds=["INBOX"], maxResults=max_results)
        .execute()
    )
    messages = result.get("messages", [])
    emails: List[Dict] = []

    for msg_meta in messages:
        msg = (
            service.users()
            .messages()
            .get(userId="me", id=msg_meta["id"], format="raw")
            .execute()
        )
        raw_data = base64.urlsafe_b64decode(msg["raw"].encode("ASCII"))
        mime_msg = email.message_from_bytes(raw_data)

        body = ""
        if mime_msg.is_multipart():
            for part in mime_msg.walk():
                if part.get_content_type() == "text/plain":
                    body += part.get_payload(decode=True).decode(errors="ignore")
        else:
            body = mime_msg.get_payload(decode=True).decode(errors="ignore")

        emails.append(
            {
                "id": msg["id"],
                "thread_id": msg["threadId"],
                "subject": mime_msg["Subject"],
                "sender": mime_msg["From"],
                "body": body.strip(),
            }
        )
    return emails

# ========= STEP 2: RAG Pipeline Setup =========

# Load your vector store
embedding = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory="db", embedding_function=embedding)

# Create retriever and chain
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# ========= STEP 3: Generate Answer =========

def generate_reply(question: str) -> str:
    result = qa_chain.invoke(question)
    return result["result"]

# ========= STEP 4: Send Reply =========

def create_message(to, subject, message_text):
    message = MIMEText(message_text)
    message['to'] = to
    message['subject'] = subject
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    return {'raw': raw}

def send_message(service, message):
    return service.users().messages().send(userId='me', body=message).execute()

# ========= STEP 5: Interactive CLI =========
def main():
    service = get_service()
    emails = get_latest_emails(service, max_results=3)

    if not emails:
        print("❌ No recent emails.")
        return

    # List choices
    print("\n📬 Latest emails:")
    for idx, em in enumerate(emails, 1):
        print(f"{idx}. {em['subject']}   [{em['sender']}]")

    # Pick one
    sel = input("\nSelect email number to open (or 'q' to quit): ").strip().lower()
    if sel in {"q", "quit"}:
        return
    try:
        idx = int(sel) - 1
        email_data = emails[idx]
    except (ValueError, IndexError):
        print("Invalid choice.")
        return

    # Show the chosen email
    print("\n" + "=" * 60)
    print(f"Subject: {email_data['subject']}")
    print(f"From   : {email_data['sender']}\n")
    print(email_data["body"][:1000] + ("..." if len(email_data["body"]) > 1000 else ""))
    print("=" * 60)

    draft = None
    while True:
        cmd = input("\nCommand [generate/send/quit]: ").strip().lower()

        if cmd == "generate":
            draft = generate_reply(email_data["body"])
            print("\n📝 Draft reply:\n" + "-" * 50 + f"\n{draft}\n" + "-" * 50)

        elif cmd == "send":
            if draft is None:
                print("No draft found, generating one now...")
                draft = generate_reply(email_data["body"])

            reply_msg = create_message(
                to=email_data["sender"],
                subject=f"Re: {email_data['subject']}",
                message_text=draft,
            )
            # keep threading intact
            reply_msg["threadId"] = email_data["thread_id"]
            send_message(service, reply_msg)
            print("✅ Reply sent!")
            break

        elif cmd in {"quit", "q"}:
            break
        else:
            print("Unknown command. Use generate, send, or quit.")

if __name__ == "__main__":
    main()
