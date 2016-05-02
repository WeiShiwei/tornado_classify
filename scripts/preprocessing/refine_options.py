#!/usr/bin/env python
# -*- coding: utf-8 -*- "
import os
import sys

CustomTabularExporter_OPTIONS = {
		"project": "",
		"format": "tsv",
		"separator": "\t",
		"lineSeparator": "\n",
		"encoding": "UTF-8",
		"outputColumnHeaders": 'true',
		"outputBlankRows": 'false',
		"columns": [
		  {
		    "name": "name",
		    "reconSettings": {
		      "output": "entity-name",
		      "blankUnmatchedCells": 'false',
		      "linkToEntityPages": 'true'
		    },
		    "dateSettings": {
		      "format": "iso-8601",
		      "useLocalTimeZone": 'false',
		      "omitTime": 'false'
		    }
		  },
		  {
		    "name": "unit",
		    "reconSettings": {
		      "output": "entity-name",
		      "blankUnmatchedCells": 'false',
		      "linkToEntityPages": 'true'
		    },
		    "dateSettings": {
		      "format": "iso-8601",
		      "useLocalTimeZone": 'false',
		      "omitTime": 'false'
		    }
		  },
		  {
		    "name": "brand",
		    "reconSettings": {
		      "output": "entity-name",
		      "blankUnmatchedCells": 'false',
		      "linkToEntityPages": 'true'
		    },
		    "dateSettings": {
		      "format": "iso-8601",
		      "useLocalTimeZone": 'false',
		      "omitTime": 'false'
		    }
		  },
		  {
		    "name": "spec",
		    "reconSettings": {
		      "output": "entity-name",
		      "blankUnmatchedCells": 'false',
		      "linkToEntityPages": 'true'
		    },
		    "dateSettings": {
		      "format": "iso-8601",
		      "useLocalTimeZone": 'false',
		      "omitTime": 'false'
		    }
		  },
		  {
		    "name": "attrs",
		    "reconSettings": {
		      "output": "entity-name",
		      "blankUnmatchedCells": 'false',
		      "linkToEntityPages": 'true'
		    },
		    "dateSettings": {
		      "format": "iso-8601",
		      "useLocalTimeZone": 'false',
		      "omitTime": 'false'
		    }
		  }
		]
}

GET_ALL_PROJECT_METADATA_URL = 'http://google-refine.gldjc.com/command/core/get-all-project-metadata'
EXPORT_URL = 'http://google-refine.gldjc.com/command/core/export-rows/'

LEARN_CORPUS_PATH = os.path.expanduser( os.path.join( '~','scikit_learn_data' ) )
TEST_CORPUS_PATH = os.path.expanduser( os.path.join( '~','scikit_test_data', 'Google-refine' ) )