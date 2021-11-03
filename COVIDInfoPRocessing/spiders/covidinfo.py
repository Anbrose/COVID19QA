import scrapy
from ..items import CovidinfoprocessingItem


class CovidinfoSpider(scrapy.Spider):
    name = 'covidinfo'
    allowed_domains = ['http://www.abc.net.au', 'https://www.abc.net.au']
    start_urls = ['http://www.abc.net.au/news/story-streams/coronavirus/']
    base = "http://www.abc.net.au/news"

    def parse(self, response):
        pass
        news_links = response.xpath('//div[@class="_29FJI"]/div/div/div/h3/span/a/@href')
        for i in news_links:
            yield scrapy.Request(url=self.base + i.get(), callback=self.handle_subDomains, dont_filter=True)

    def handle_subDomains(self, response):
        items = CovidinfoprocessingItem()
        bodies = response.xpath('//div[@id="body"]/div/div[1]/div/div/p')
        print("-" * 80)
        print(len(bodies))
        print("-" * 80)
        for body in bodies:
            items['raw_text'] = body.get()
            yield items




