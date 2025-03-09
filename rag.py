import logging
from math import ceil

import pandas as pd
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from mistralai import Mistral
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import os
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english')) & set(stopwords.words('russian'))

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
MISTRAL_API_ENDPOINT = os.getenv("MISTRAL_API_ENDPOINT")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = TELEGRAM_BOT_TOKEN

MISTRAL_API_ENDPOINT = MISTRAL_API_ENDPOINT
MISTRAL_MODEL = 'mistral-medium'
MISTRAL_CLIENT = Mistral(api_key=MISTRAL_API_ENDPOINT)

PINECONE_API_KEY = PINECONE_API_KEY
PINECONE_ENVIRONMENT = 'us-east-1'
INDEX_NAME = 'similar-products'

pc = Pinecone(api_key=PINECONE_API_KEY)

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        'Hi! I am your Mistral RAG Recommendation bot. Send me a message and I will retrieve and generate a response for you.')


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_message = update.message.text
    logger.info(f"Received message: {user_message}")

    retrieved_docs, retrieved_names = retrieve_documents(user_message)

    response = generate_response(user_message, retrieved_docs, retrieved_names)

    await update.message.reply_text(response)


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s.!?-]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)


def retrieve_documents(query: str):

    query_vector = model.encode(preprocess_text(query)).tolist()

    index = pc.Index(INDEX_NAME)
    results = index.query(vector=query_vector, top_k=5, include_metadata=True)
    logger.info(f"Retrieved results: {results}")

    retrieved_docs = []
    retrieved_names = []
    for match in results['matches']:
        doc_id = match['id']
        try:
            retrieved_docs.append(f"{match['metadata']['name']}: {match['metadata']['desc']}")
            retrieved_names.append(match['metadata']['name'])
        except KeyError:
            logger.warning(f"Document ID {doc_id} not found in local dictionary.")
    # print(retrieved_names)
    # print(retrieved_docs)
    return retrieved_docs, retrieved_names


def generate_response(query: str, documents: list, names: list):
    # prompt = f"""
    # Just write in number list it: {names} and add some short description for each product from it: {documents}!
    # Please, write the answer only in the language of the user's question: {query}!
    # """
    prompt = f"""
    You are an AI assistant designed to recommend some similar products to people.
    We give you the best recommendations.
    Just write in number list it: {names} and add some short description for each product from it: {documents}!
    You need to output them in the language in which the question was asked!!!
    Here is the user's question:
    [{query}]

    Please, write the answer only in the language of the user's question.
        """
    try:
        response = MISTRAL_CLIENT.chat.complete(
            model=MISTRAL_MODEL,
            messages=[{'role': "user", 'content': prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "Sorry, I couldn't generate a response at the moment."


def generate_embeddings(all_data, name_desc_dict):
    embeddings = []
    for ind, name, desc, prep_desc in all_data:
        name_desc_dict.append({'name': name, 'desc': desc})
        embedding = model.encode(prep_desc)
        embeddings.append({'id': ind, 'values': embedding.tolist(), 'metadata': {'name': name, 'desc': desc}})

    return embeddings


def main() -> None:
    data_folder = './data'
    data = pd.read_csv(os.path.join(data_folder, 'concat_data.csv'))
    data_numpy = data.to_numpy()
    data_len = len(data)
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
    # Max 2MB
    index = pc.Index(INDEX_NAME)
    stats = index.describe_index_stats()
    print(f"Всего векторов в индексе: {stats['total_vector_count']}")
    vector_db_len = stats['total_vector_count']
    if vector_db_len < data_len:
        name_desc_dict = []
        embeddings = generate_embeddings(data_numpy, name_desc_dict)
        len_emb = len(embeddings)
        batch_size = 1000
        nums_of_batches = ceil(len_emb / batch_size)
        for i in range(nums_of_batches):
            st = i * batch_size
            fin = (i + 1) * batch_size if i < nums_of_batches - 1 else len_emb
            index.upsert(embeddings[st:fin])

    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.run_polling()


if __name__ == '__main__':
    main()
