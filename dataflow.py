import os
import json
import re

from bytewax.dataflow import Dataflow
from bytewax.inputs import ManualInputConfig, distribute
from bytewax.execution import run_main
from bytewax.outputs import StdOutputConfig

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, AutoModelForSeq2SeqLM

from websocket import create_connection

WEBHOOK_URL = os.getenv("WEBHOOK_URL")
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

ticker_list = ["*"]


def input_builder(worker_index, worker_count, resume_state):
    state = resume_state or None
    worker_tickers = list(distribute(ticker_list, worker_index, worker_count))
    print({"subscribing to": worker_tickers})

    def news_input(worker_tickers, state):
        ws = create_connection("wss://stream.data.alpaca.markets/v1beta1/news")
        print(ws.recv())
        ws.send(json.dumps({"action":"auth","key":f"{API_KEY}","secret":f"{API_SECRET}"}))
        print(ws.recv())
        ws.send(json.dumps({"action":"subscribe","news":worker_tickers}))
        print(ws.recv())
        
        while True:
        # articles = [{"T":"n","id":31248067,"headline":"Tesla Vehicles Could Be Banned From Leaving During A Hurricane In This State","summary":"A lawmaker in one American state could make it hard for owners of electric vehicles to get out of the state in the event of a hurricane. Here’s the potential law and why it’s important.","author":"Chris Katje","created_at":"2023-03-07T22:58:40Z","updated_at":"2023-03-07T22:58:40Z","url":"https://www.benzinga.com/news/23/03/31248067/tesla-vehicles-could-be-banned-from-leaving-during-a-hurricane-in-this-state","content":"\u003cp\u003eA lawmaker in one American state could make it hard for owners of electric vehicles to get out of the state in the event of a hurricane. Here\u0026rsquo;s the potential law and why it\u0026rsquo;s important.\u003c/p\u003e\r\n\r\n\u003cp\u003e\u003cstrong\u003eWhat Happened:\u003c/strong\u003e States have passed laws aimed at banning the sale of gas-powered vehicles in the future. One state took it a step further by seeking to ban electric vehicle \u003ca href=\"https://www.benzinga.com/news/23/01/30424292/taking-on-elon-musk-this-state-legislature-could-ban-electric-vehicle-sales-by-2035\"\u003esales in the future.\u003c/a\u003e\u003c/p\u003e\r\n\r\n\u003cp\u003eOne of the leading states for electric vehicle purchases could now see a temporary ban on using electric vehicles during the time of a crisis.\u003c/p\u003e\r\n\r\n\u003cp\u003eFlorida Republican state Sen.\u0026nbsp;\u003cstrong\u003eJonathan Martin\u003c/strong\u003e is considering legislation to ban electric vehicles like those from \u003cstrong\u003eTesla Inc\u003c/strong\u003e (NASDAQ:\u003ca class=\"ticker\" href=\"https://www.benzinga.com/stock/TSLA#NASDAQ\"\u003eTSLA\u003c/a\u003e) to be used during hurricane evacuations in the state, according to \u003ca href=\"https://electrek.co/2023/03/06/florida-lawmaker-wants-to-ban-evs-from-hurricane-evacuations/\"\u003eElectrek\u003c/a\u003e.\u0026nbsp;\u003c/p\u003e\r\n\r\n\u003cp\u003eMartin told the state\u0026rsquo;s Department of Transportation that electric vehicles could block traffic during evacuations if they run out of battery charge.\u003c/p\u003e\r\n\r\n\u003cp\u003eMartin serves on the Committee on Environment and Natural Resources and the Select Committee on Resiliency.\u003c/p\u003e\r\n\r\n\u003cp\u003eThe Select Committee on Resiliency met with the Florida Department of Transportation executive director of transportation technologies in Florida.\u003c/p\u003e\r\n\r\n\u003cp\u003eAmong the topics discussed were the $198 million the state is going to get from the Bipartisan Infrastructure Law for electric vehicle charging infrastructure from the current administration led by \u003cstrong\u003ePresident Joe Biden.\u003c/strong\u003e\u003c/p\u003e\r\n\r\n\u003cp\u003eThe legislation requires electric vehicle charging stations to be 50 miles apart and serve all electric vehicles.\u003c/p\u003e\r\n\r\n\u003cp\u003e\u0026ldquo;With a couple of guys behind you, you can\u0026rsquo;t get out of the car and push it to the side of the road. Traffic backs up. And what might look like a two-hour trip might turn into an eight-hour trip once you\u0026rsquo;re on the road,\u0026rdquo; Martin said.\u003c/p\u003e\r\n\r\n\u003cp\u003eMartin said his concern is with the electric vehicle infrastructure available in the state of Florida.\u003c/p\u003e\r\n\r\n\u003cp\u003e\u003cem\u003eRelated Link: \u003ca href=\"https://www.benzinga.com/trading-ideas/22/06/27568560/4-stocks-to-watch-this-hurricane-season\"\u003e4 Stocks To Watch This Hurricane Season\u0026nbsp;\u003c/a\u003e\u003c/em\u003e\u003c/p\u003e\r\n\r\n\u003cp\u003e\u003cstrong\u003eWhy It\u0026rsquo;s Important:\u003c/strong\u003e The Florida Department of Transportation told Martin it isn\u0026rsquo;t a fan of banning electric vehicles during hurricane evacuations and that it is looking into portable EV chargers.\u003c/p\u003e\r\n\r\n\u003cp\u003e\u0026ldquo;We have our emergency assistance vehicles that we deploy during a hurricane evacuation that have gas \u0026hellip; we need to provide that same level of service to electrical vehicles,\u0026rdquo; Department of Transportation director of transportation technologies \u003cstrong\u003eTrey Tillander \u003c/strong\u003esaid.\u003c/p\u003e\r\n\r\n\u003cp\u003eThe Tampa Bay Times \u003ca href=\"https://www.tampabay.com/hurricane/2023/02/24/florida-lawmaker-suggests-limiting-electric-vehicles-during-hurricane-evacuations/\"\u003ereported\u003c/a\u003e\u0026nbsp;around 1% of the vehicles in Florida are electric vehicles. One of the owners of an EV is state Sen.\u0026nbsp;\u003cstrong\u003eTina Polsky.\u003c/strong\u003e\u003c/p\u003e\r\n\r\n\u003cp\u003e\u0026ldquo;I don\u0026rsquo;t think you can ban an electric vehicle from evacuating because that may be the only car someone has,\u0026rdquo; Polsky said.\u003c/p\u003e\r\n\r\n\u003cp\u003eIn December 2022, there were 203,094 electric vehicles registered in the state of Florida.\u003c/p\u003e\r\n\r\n\u003cp\u003eThe increased funding for charging infrastructure could help ease concerns over charging.\u003c/p\u003e\r\n\r\n\u003cp\u003eUltimately, once people are on the road headed out of the state, they likely won\u0026rsquo;t be able to stop at a charging station, similar to people not being able to quickly stop at a gas station.\u003c/p\u003e\r\n\r\n\u003cp\u003eJust like people prepare for the evacuation by filling up their vehicle with gas, owners of electric vehicles will likely need to fully charge their vehicle before evacuating the state.\u003c/p\u003e\r\n\r\n\u003cp\u003eThe comments from the state senator may have Florida residents thinking about owning at least one non-electric vehicle or a hybrid to ensure they have the best chance to exit the state without future restrictions and without the potential of running out of charge and not finding stations prevalent.\u003c/p\u003e\r\n\r\n\u003cp\u003e\u003cem\u003eRead Next:\u0026nbsp;\u003ca href=\"https://www.benzinga.com/analyst-ratings/analyst-color/23/03/31172188/tesla-analysts-praise-vertical-integration-after-investor-day-but-want-more-from-el\"\u003eTesla Analysts Praise Vertical Integration After Investor Day, But Want More From Elon Musk: \u0026#39;Long On Vision, Short On Specifics\u0026#39;\u003c/a\u003e\u003c/em\u003e\u003c/p\u003e\r\n\r\n\u003cp\u003e\u003cem\u003ePhoto:\u0026nbsp;\u003ca href=\"https://www.shutterstock.com/g/hsaduraphotos\"\u003eHenryk Sadura\u003c/a\u003e\u0026nbsp;via Shutterstock\u003c/em\u003e\u003c/p\u003e\r\n\r\n\u003cp\u003e\u003cbr /\u003e\r\n\u0026nbsp;\u003c/p\u003e\r\n","symbols":["TSLA"],"source":"benzinga"}]
            articles = json.loads(ws.recv())
            for article in articles:
                yield state, (article["source"], article)

    return news_input(worker_tickers, state)


flow = Dataflow()
flow.input("inp", ManualInputConfig(input_builder))
flow.inspect(print)

def update_articles(articles, news):
    if news['id'] in articles:
        news['update'] = True
    else:
        articles.append(news['id'])
        news['update'] = False
    return articles, news

flow.stateful_map("source_articles", lambda: list(), update_articles)

flow.filter(lambda x: not x[1]['update'])

sent_tokenizer = AutoTokenizer.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis")
sent_model = AutoModelForSequenceClassification.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis")
sent_nlp = pipeline("sentiment-analysis", model=sent_model, tokenizer=sent_tokenizer)

def sentiment_analysis(ticker__news):
    ticker, news = ticker__news
    sentiment = sent_nlp([news["headline"]])
    news['sentiment'] = sentiment[0]
    print(sentiment[0])
    return (ticker, news)

flow.map(sentiment_analysis)

# Let's create a pipeline
sum_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
sum_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
summarizer = pipeline("summarization", tokenizer=sum_tokenizer, model=sum_model)
tag_re = re.compile(r'(<!--.*?-->|<[^>]*>)')

def summarize(ticker__news):
    ticker, news = ticker__news
    article = news['content']
    article_no_tags = tag_re.sub('', article)
    article_no_tags = article_no_tags.replace("\r", "").replace("\n", "")
    summary = summarizer(article_no_tags, max_length=130, min_length=30, do_sample=False)
    news['bart_summary'] = summary[0]['summary_text']
    print(summary[0]['summary_text'])
    return (ticker, news)

flow.map(summarize)

flow.capture(StdOutputConfig())

if __name__ == '__main__':
    run_main(flow)