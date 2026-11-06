from dotenv import load_dotenv

from .cluster import LAB_ROOT
from .own_agent import OwnAgentSession


def main():
    load_dotenv(LAB_ROOT / '.env')
    with OwnAgentSession() as session:
        print(f'Session: {session.session_id}')
        while True:
            try:
                question = input('> ').strip()
            except (EOFError, KeyboardInterrupt):
                break
            if question == '/exit':
                break
            if question:
                result = session.ask(question)
                print(result['answer'])
                print(result['trace_url'])


if __name__ == '__main__':
    main()
