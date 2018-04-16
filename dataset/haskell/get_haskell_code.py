import os


code_dirs = ['ghc/']

# Total files read  - 
files_read = 0

with open('haskell_code.txt', 'a', encoding='utf-8') as f:
    for code_dir in code_dirs:
        print('Code Directory:', code_dir)
        for root, dirs, files in os.walk(code_dir):
            for file in files:
                files_read += 1
                if file.split('.')[-1] == 'hs':
                    try:
                        with open(os.path.join(root, file), 'r', encoding='utf-8') as code_file:
                            content = code_file.read()
                        f.write(str(content))
                        f.write('\n\n')
                        print('Total Files Read:', files_read)
                    except Exception as e:
                        print(str(e))
