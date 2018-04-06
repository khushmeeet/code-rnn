import os


code_dirs = ['/Users/khushmeetsingh/anaconda/lib/python3.6',
             '/Users/khushmeetsingh/anaconda/lib/python3.6/site-packages']

files_read = 0

with open('python_code.txt', 'a', encoding='utf-8') as file:
    for code_dir in code_dirs:
        for root, dirs, files in os.walk(code_dir):
            for file in files:
                files_read += 1
                if '.py' in file:
                    try:
                        with open(os.path.join(root, file), 'r') as code_file:
                            content = code_file.read()
                        file.write(content)
                        file.write('\n\n')
                    except Exception as e:
                        print(str(e))