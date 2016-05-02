#!/usr/bin/python
# -*- coding: utf-8 -*-

import os,sys
import re
reload(sys)
sys.setdefaultencoding('utf8')

import jieba
jieba.load_userdict(os.path.join( os.path.abspath(os.path.dirname(__file__)) , 'userdict.txt'))

class Segment(object):
	"""docstring for Segment"""
	def __init__(self, arg):
		super(Segment, self).__init__()
		self.arg = arg
	
	@classmethod
	def cut(self,sentence):
		return ' '.join( jieba.cut(sentence) )
	@classmethod
	def seg(self,docs):
		docs_seg = list()
		for doc in docs:
			docs_seg.append(' '.join( jieba.cut(doc) ))
		return docs_seg



class Feature_Engineering(object):
    """docstring for Feature_Engineering"""
    prog = re.compile(r'(?u)\b\w\w+\b') # 单个的unicode字符是不会要的
    # prog = re.compile(r'(?u)\b[\w-]{2,}\b')
    
    @classmethod
    def normalize(self,line):
        """
        1. 全部大写处理
        sklearn.feature_extraction.text.TfidfVectorizer有默认参数，所以似乎没有这个必要
        lowercase : boolean, default True
        Convert all characters to lowercase before tokenizing.
        
        2. \* 规范化处理中文全角到英文半角,例如'（' => '(' or '：'=>':' *\
        3. prog = re.compile(r'(?u)\b\w\w+\b')该正则表达式的作用
        4. 删除纯数字串，保留（数字字符、英文字符(必须要有一个英文字符)、-、/）等型号字符的组合
        5. 规范化计量单位，或者删除计量单位
        6. 分词处理
        7. 删除单个unicode非中文字符：Ⅰ Ⅱ
        """
    	line = unicode(line).upper() # 全部大写处理
        ## 规范化处理，中文全角到英文半角,例如'（' => '(' or '：'=>':'
        # print line
    	line = Feature_Engineering.__tokenize(line);#print line
    	line = Feature_Engineering.__pureDigits_dicard(line);#print line
    	line = Segment.cut(line);#print line
        line = Feature_Engineering.__pureDigits_dicard(line);#print line # 'Φ8'
    	return line

    @classmethod
    def normalize_lines(self,lines):
        for i in xrange(len(lines)):
            line = unicode(lines[i])
            lines[i] = Feature_Engineering.normalize(line)

        return lines

    @classmethod
    def __tokenize(self,line):
        # import pdb;pdb.set_trace()
        res = Feature_Engineering.prog.findall(unicode(line))
        return ' '.join(res)

    @classmethod
    def __pureDigits_dicard(self,line):
        # line = unicode(line)
        temp = list()
        for word in line.split():
            if word.isdigit():
                continue
            temp.append(word)
        return ' '.join(temp)


def main():
    # s = unicode('铜芯聚氯乙烯绝缘软电线		天河天虹	450/750V  10m㎡  结构:49/0.52  ZB-BVR')#;s = unicode('HRB400螺纹钢   φ10')
    s = unicode("""电缆防火涂料,      
        20kg/桶
        涂布率约㎡:3-5kg/㎡      
        YA-G型   
        规格参数:规格：20kg/桶 参数：1㎡/0.5mm/1Kg-0.8mm 
        GB14907-2002""")
    s = unicode('原木 立方米 鹏辉 水曲柳')
    s = unicode('原木 立方米 鹏辉 水煮柚')
    s = unicode('水曲柳')
    print Feature_Engineering.normalize(s)

    # 铜芯聚氯乙烯绝缘软电线     天河天虹    450/750V  10m㎡  结构:49/0.52  ZB-BVR
    # 铜芯聚氯乙烯绝缘软电线 天河天虹 450 750V 10m 结构 49 52 ZB BVR
    # 铜芯聚氯乙烯绝缘软电线 天河天虹 750V 10m 结构 ZB BVR
    # 铜芯聚氯乙烯绝缘软电线   天河 天虹   750V   10m   结构   ZB   BVR

    # print Segment.cut(s)
    # 铜芯聚氯乙烯绝缘软电线          天河 天虹   450 / 750V     10m ㎡     结构 : 49 / 0.52     ZB - BVR


if __name__ == "__main__":
	main()