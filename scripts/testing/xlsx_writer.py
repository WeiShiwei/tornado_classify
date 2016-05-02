#!/usr/bin/python
# coding:utf-8

import xlsxwriter

class XlsxWriter(object):
	"""docstring for XlsxWriter"""
	
	def __init__(self, filepath):
		super(XlsxWriter, self).__init__()
		self.workbook = xlsxwriter.Workbook( filepath )
		
		# Add a bold format to use to highlight cells.
		self.bold = self.workbook.add_format({'bold': True})
		self.red_font_format = self.workbook.add_format() 
		self.red_font_format.set_font_color('red')

		self._worksheet1 = self.workbook.add_worksheet()
		self._worksheet2 = self.workbook.add_worksheet()
		self._worksheet3 = self.workbook.add_worksheet()

		self.worksheets = {
			"measure":self._worksheet1,
			"documents":self._worksheet2,
			"visual":self._worksheet3
		}

	def __del__(self):
		self.workbook.close()
 	
 	def write_meature_worksheet(self, json_result):
		worksheet = self.worksheets['measure']

		# Write some data headers.
		# worksheet.write('A1', 'Measure', self.bold)
		# worksheet.write('B1', 'Value', self.bold)


		# Some data we want to write to the worksheet.
		test_report = (
			['accuracy_score', json_result.get('accuracy_score', '')],
			['recall_score',   json_result.get('recall_score', '')],
			['f1_score',  json_result.get('f1_score', '')],
			# ['classification_report',    json_result.get('classification_report', '')],
			# ['confusion_matrix',    json_result.get('confusion_matrix', '')]
		)

		# Start from the first cell below the headers.
		row = 1
		col = 0

		# Iterate over the data and write it out row by row.
		# for measure, value in (test_report):
		# 	worksheet.write(row, col,     measure)
		# 	worksheet.write(row, col + 1, value)
		# 	row += 1
		accuracy_recall_f1_score = '\n'.join( ['accuracy_score: '+ json_result.get('accuracy_score', ''),
												'recall_score: '+ json_result.get('recall_score', ''),
												'f1_score: '+ json_result.get('f1_score', ''),
		])
		worksheet.insert_textbox('A1', accuracy_recall_f1_score,
                         {'width': 288, 'height': 100} )
		worksheet.insert_textbox('A5', 'classification_report: \n'+json_result.get('classification_report', ''),
                         {'width': 288, 'height': 500} )
		worksheet.insert_textbox('F5', 'confusion_matrix: \n'+json_result.get('confusion_matrix', ''),
                         {'width': 288, 'height': 500})

	def write_documents_worksheet(self, json_result):
		worksheet = self.worksheets['documents']

		# Write some data headers.
		worksheet.write('A1', 'ytest', self.bold)
		worksheet.write('B1', 'ypred', self.bold)
		worksheet.write('C1', 'boolean_rules_pred', self.bold)
		worksheet.write('D1', 'classifiers_pred', self.bold)
		worksheet.write('E1', 'multi_patterns', self.bold)
		worksheet.write('F1', 'document', self.bold)

		# Some data we want to write to the worksheet.
		predict_documents = json_result.get("predict_documents", list())

		# Start from the first cell below the headers.
		row = 1
		col = 0

		# import pdb;pdb.set_trace()
		# Iterate over the data and write it out row by row.
		for ytest,ypred,boolean_rules_pred,classifiers_pred,multi_patterns,document in predict_documents:
			if ytest == ypred:
				worksheet.write(row, col,     ytest)
				worksheet.write(row, col + 1, ypred)
				worksheet.write(row, col + 2, boolean_rules_pred)
				worksheet.write(row, col + 3, classifiers_pred)
				worksheet.write(row, col + 4, multi_patterns)
				worksheet.write(row, col + 5, document)
			else:
				worksheet.write(row, col,     ytest, self.red_font_format)
				worksheet.write(row, col + 1, ypred, self.red_font_format)
				worksheet.write(row, col + 2, boolean_rules_pred, self.red_font_format)
				worksheet.write(row, col + 3, classifiers_pred, self.red_font_format)
				worksheet.write(row, col + 4, multi_patterns, self.red_font_format)
				worksheet.write(row, col + 5, document, self.red_font_format)
			row += 1



	def write(self, json_result):
		# import pdb;pdb.set_trace()
		self.write_meature_worksheet( json_result )
		self.write_documents_worksheet( json_result )
		# self.write_meature_worksheet( json_result )

		# self.workbook.close()