import datetime
import sys
import token
import tokenize


def do_file(fname):
    source = open(fname)
    mod = open(fname + "_processed", "w")

    prev_toktype = token.INDENT
    first_line = None
    last_lineno = -1
    last_col = 0

    tokgen = tokenize.generate_tokens(source.readline)
    for toktype, ttext, (slineno, scol), (elineno, ecol), ltext in tokgen:
        if 0:
            print("%10s %-14s %-20r %r" % (
                tokenize.tok_name.get(toktype, toktype),
                "%d.%d-%d.%d" % (slineno, scol, elineno, ecol),
                ttext, ltext
            ))
        if slineno > last_lineno:
            last_col = 0
        if scol > last_col:
            mod.write(' ' * (scol - last_col))
        if toktype == token.STRING and prev_toktype == token.INDENT:
            mod.write('')
        elif toktype == tokenize.COMMENT:
            mod.write('')
        else:
            mod.write(ttext)
        prev_toktype = toktype
        last_col = ecol
        last_lineno = elineno

print('Working')
start = datetime.datetime.second
do_file('python_code.txt')
end = datetime.datetime.second
print('Done!')
print('Time Taken:', end-start)
