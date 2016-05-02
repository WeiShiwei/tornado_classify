#!/usr/bin/python
# -*- coding: utf-8 -*-

import os,sys
RIGHT_SYMBOL = '@'

class Trie(object):
    """docstring for Trie"""
    
    def __init__(self):
        super(Trie, self).__init__()
        self.trie = self.__construct_trie(os.path.join( os.path.abspath(os.path.dirname(__file__)) , 'trie.repo'))

    def __construct_trie(self, trie_file):
        """ construct trie
        """
        trie = dict()
        with open(trie_file,'rb') as f:
            lineno = 0 
            for line in f.read().rstrip().decode('utf-8').split('\n'):
                lineno += 1
                try:
                    word,repl = line.split()
                    word += RIGHT_SYMBOL ### 右边界
                    p = trie
                    for c in word:
                        if c not in p:
                            p[c] = {}
                        p = p[c]
                    p[''] = repl
                except ValueError, e:
                    print '%s at line %s %s' % (trie_file,  lineno, line)
                    raise ValueError, e
        return trie
    

    def find(self, sentence):
        """ 
        """
        sentence = sentence.decode('utf-8')
        length = len(sentence) 
        trie = self.trie
        res_str = ''

        i = 0
        while i < length:
            p = trie
            if p.get(sentence[i]) == None :#或者是空白符
                res_str += sentence[i]
                i += 1
                continue
            else:
                j=i
                p = p.get(sentence[j])
                while p!=None : 
                    if p.get('') != None:
                        break                    
                    j += 1
                    if j >= length:
                        break
                    else:
                        p = p.get(sentence[j])

                # 下面三个判断的顺序不可改变
                if p == None:
                    res_str += sentence[i]
                    i += 1
                    continue
                if p.get('') != None:
                    return p.get('')
                    res_str += p.get('')
                    i = j + 1
                    continue
                if j == length:
                    res_str += sentence[i:]
                    i = length
                    continue
        return None



def main():
    print '='*50
    print "Trie trie.repo loading:"
    trie_file = os.path.join( os.path.abspath(os.path.dirname(__file__)) , 'trie.repo')

    trie = Trie()
    res = trie.find('No one can make you feel inferior without your consent. NHBVV@ ***')
    print res


if __name__ == "__main__":
    main()