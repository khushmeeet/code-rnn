import os


code_dirs = ['/Users/khushmeetsingh/anaconda/lib/python3.6',
             '/Users/khushmeetsingh/anaconda/lib/python3.6/site-packages']

# Total files read  - 143993
files_read = 0

with open('python_code.txt', 'a', encoding='utf-8') as f:
    for code_dir in code_dirs:
        print('Code Directory:', code_dir)
        for root, dirs, files in os.walk(code_dir):
            for file in files:
                files_read += 1
                if file.split('.')[-1] == 'py':
                    try:
                        with open(os.path.join(root, file), 'r', encoding='utf-8') as code_file:
                            content = code_file.read()
                        f.write(str(content))
                        f.write('\n\n')
                        print('Total Files Read:', files_read)
                    except Exception as e:
                        print(str(e))
