from bs4 import BeautifulSoup
import requests
from lxml import etree
from datetime import datetime, timedelta


def get_weather(url):
    response = requests.get(url)
    response.encoding = "utf-8"
    html_doc = response.text
    # 解析HTML
    tree = etree.HTML(html_doc)
    # 获取今天的日期
    today = datetime.now().date()

    results = []
    day_weather = ""
    for index, li in  enumerate(tree.xpath("//ul[contains(@class, 't')]/li[contains(@class, 'sky')]")):
        day_weather += "日期：" + (today + timedelta(days=index)).strftime('%Y-%m-%d')
        day_weather += "，天气：" + "".join([ele.strip() for ele in li.xpath("./p[@class='wea']//text()")])
        day_weather += "，气温：" + " ".join([ele.strip() for ele in li.xpath("./p[@class='tem']//text()")])
        results.append(day_weather)
        day_weather = ""

    return results



if __name__ == '__main__':
    url = "http://www.weather.com.cn/weather/101020100.shtml"
    weather = get_weather(url)
    print(weather)