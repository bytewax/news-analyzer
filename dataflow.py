import os
import json

from bytewax.dataflow import Dataflow
from bytewax.inputs import ManualInputConfig, distribute
from bytewax.execution import run_main
from bytewax.outputs import StdOutputConfig

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, PegasusTokenizer, PegasusForConditionalGeneration

from websocket import create_connection

WEBHOOK_URL = os.getenv("WEBHOOK_URL")
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

ticker_list = ["AAPL", "MSFT", "AMZN"]


def input_builder(worker_index, worker_count, resume_state):
    state = resume_state or None
    worker_tickers = list(distribute(ticker_list, worker_index, worker_count))
    print({"subscribing to": worker_tickers})

    def yf_input(worker_tickers, state):
        ws = create_connection("wss://stream.data.alpaca.markets/v1beta1/news")
        ws.send(json.dumps({"action":"auth","key":f"{API_KEY}","secret":f"{API_SECRET}"}))
        ws.send(json.dumps({"action":"subscribe","news":worker_tickers}))
        while True:
            yield state, ws.recv()

    return yf_input(worker_tickers, state)


flow = Dataflow()
flow.input("inp", ManualInputConfig(input_builder))
flow.inspect(print)

sent_tokenizer = AutoTokenizer.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis")
sent_model = AutoModelForSequenceClassification.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis")
sent_nlp = pipeline("sentiment-analysis", model=sent_model, tokenizer=sent_tokenizer)

def sentiment_analysis(ticker__news):
    ticker, news = ticker__news
    sent_nlp([news["headline"]])
    news['sentiment'] = sent_nlp[0]
    return (ticker, news)

flow.map(sentiment_analysis)

# Let's load the model and the tokenizer 
model_name = "human-centered-summarization/financial-summarization-pegasus"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

def summarize(ticker__news):
    ticker, news = ticker__news
    input_ids = tokenizer(news["content"], return_tensors="pt").input_ids

    # Generate the output (Here, we use beam search but you can also use any other strategy you like)
    output = model.generate(
        input_ids, 
        max_length=32, 
        num_beams=5, 
        early_stopping=True
    )
    news['pegasus_summary'] = tokenizer.decode(output[0], skip_special_tokens=True)
    return (ticker, news)

flow.map(summarize)

flow.capture(StdOutputConfig())

if __name__ == '__main__':
    run_main(flow)