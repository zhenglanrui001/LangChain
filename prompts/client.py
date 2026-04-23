from langserve import RemoteRunnable

if __name__ == '__main__':
    client = RemoteRunnable('http://localhost:8000/translate')
    print(client.invoke({
        'language': 'italian',
        'text': '你好'
    }))