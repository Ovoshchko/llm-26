import argparse
import json

from .session import OpenCodeSession


def main():
    parser = argparse.ArgumentParser(description='Отдельный агент OpenCode для lab3')
    parser.add_argument('question')
    parser.add_argument('--session')
    arguments = parser.parse_args()
    session = OpenCodeSession(session_id=arguments.session)
    result = session.ask(arguments.question)
    print(json.dumps({key: value for key, value in result.items() if key != 'events'},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
