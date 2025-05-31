
import newspaper
url = "https://www.indiatoday.in/fact-check/story/fact-check-2018-video-police-raid-kanpur-builder-not-bsf-soldiers-2732057-2025-05-28"
article = newspaper.Article(url)
article.download()
article.parse()
# print(article.publish_date)
# print(article.meta_description)
article.nlp()
print(article.keywords)
