# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
from bs4 import BeautifulSoup
import re

class CovidinfoprocessingPipeline:
    def process_item(self, item, spider):
        # as we launched in terminal, return the whole body might be massed
        with open("E:\\UNSWMPHIL\COVIDInfoPRocessing\COVIDInfoPRocessing\data\sample.txt", 'ab') as f:
            context = BeautifulSoup(item['raw_text']).p.string
            if context != None:
                f.write(context.encode('GB18030'))
        return None

