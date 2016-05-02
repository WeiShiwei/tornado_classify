# coding:utf-8
"""
"∨" 表示"或" （逻辑加法）
"∧" 表示"与" （逻辑乘法）
"┐" 表示"非" （逻辑否定）


规则字符串不允许括号的嵌套
example:
IF w1 THEN 
IF w1∧w2 THEN 
IF w1∧w2∧w3 THEN 
IF w1∨w2 THEN
IF w1∨w2∨w3 THEN
IF w1∧(┐w2) THEN 或者 IF w1∧┐w2 THEN ### Unfinished
IF w1∧(w2∨w3∨w4) THEN  

"""
import os
import sys
import itertools

LOGICAL_AND_SYMBOL = '∧'
LOGICAL_OR_SYMBOL = '∨'
LOGICAL_NOT_SYMBOL	= '┐'

class BooleanRuleParser(object):
	"""docstring for BooleanRuleParser"""
	
	def __init__(self, arg):
		super(BooleanRuleParser, self).__init__()
		self.arg = arg
	
	@classmethod
	def parse(self, rule_desc):
		subrules = list()
		for sub in rule_desc.split( LOGICAL_AND_SYMBOL ):
			sub = sub.strip('\t\n() ')
			subrules.append( sub.split( LOGICAL_OR_SYMBOL ))

		result = list()
		for x in itertools.product(*subrules):
			result.append('∧'.join( sorted(x)))
			# print '∧'.join( sorted(x))
		return result

def main():
	rule_desc = '三通∧(PB∨PE∨PP∨PPR∨PP-R∨PVC∨PVC-U)'
	rule_desc = '(标准块∨万能块∨斜角块∨配套块∨圈梁块∨隔墙块)∧(强度等级∨MPa)'
	rule_desc = '标准块'
	result = BooleanRuleParser.parse(rule_desc)
	print '\n'.join(result)

	# 强度等级∧标准块
	# MPa∧标准块
	# 万能块∧强度等级
	# MPa∧万能块
	# 强度等级∧斜角块
	# MPa∧斜角块
	# 强度等级∧配套块
	# MPa∧配套块
	# 圈梁块∧强度等级
	# MPa∧圈梁块
	# 强度等级∧隔墙块
	# MPa∧隔墙块

if __name__ == '__main__':
	main()