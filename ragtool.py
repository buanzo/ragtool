#!/usr/bin/env python3
# Author: Arturo "Buanzo" Busleiman https://github.com/buanzo/ragtool
#
# I code this script in 2023, but I just got to releasing it. It works with current langchain, though.
# Hope you like it. The idea was to have a simple, quick, CLI tool for RAG testing. Its very satisfying :)

import os
import argparse
import psycopg2
import json
import sys

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector, DistanceStrategy
from langchain.chains import RetrievalQA

# TODO: ENH: JSONLoader is not used currently
# from langchain.document_loaders.json_loader import JSONLoader
from langchain.document_loaders.text import TextLoader


def get_pgsql_connection_string(env_var_name='RAGTOOL_PGSQL_CONNECTION_STRING'):
    return os.environ[env_var_name]


def enable_pgvector_extension(connection_string):
    try:
        conn = psycopg2.connect(connection_string)
        cursor = conn.cursor()

        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()

        print("pgvector extension has been enabled.")

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"An error occurred: {e}")


def check_pgvector_extension(connection_string):
    try:
        conn = psycopg2.connect(connection_string)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
        extension = cursor.fetchone()

        if extension:
            print(f"pgvector is installed with version: {extension[2]}")
        else:
            print("pgvector extension is not installed.")

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"An error occurred: {e}")

def create_pgvector_collection(connection_string, collection_name, source_dir, recreate):
    embeddings = OpenAIEmbeddings()
    # Use DirectoryLoader to load all HTML files from the temp directory
    loader = DirectoryLoader(source_dir, glob="**/*", show_progress=True, loader_cls=TextLoader)
    #loader = DirectoryLoader(source_dir, glob="**/*", show_progress=True, loader_cls=JSONLoader, loader_kwargs = {'jq_schema':'.content'})
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    # Set up PGVector
    COLLECTION_NAME = collection_name
    CONNECTION_STRING = connection_string

    db = PGVector.from_documents(
        embedding=embeddings,
        documents=docs,
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=recreate,
        distance_strategy = DistanceStrategy.COSINE,
    )

def delete_pgvector_collection(connection_string, collection_name, force=False):
    try:
        # Establish a connection to the database
        conn = psycopg2.connect(connection_string)
        cursor = conn.cursor()
        
        # Fetch the UUID of the collection
        cursor.execute("SELECT uuid FROM langchain_pg_collection WHERE name = %s;", (collection_name,))
        uuid_tuple = cursor.fetchone()
        
        if uuid_tuple:
            uuid = uuid_tuple[0]
        else:
            print("Collection not found.")
            return
        
        # If 'force' flag is not set, ask for user confirmation
        if not force:
            confirm = input(f"Are you sure you want to delete the collection '{collection_name}'? [y/N]: ")
            if confirm.lower() != 'y':
                print("Aborted.")
                return
        
        # Delete the collection
        cursor.execute("DELETE FROM langchain_pg_embedding WHERE collection_id = %s;", (uuid,))
        cursor.execute("DELETE FROM langchain_pg_collection WHERE uuid = %s;", (uuid,))
        
        # Commit the changes
        conn.commit()
        print(f"Collection '{collection_name}' has been deleted.")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"An error occurred: {e}")

def query_pgvector_collection(connection_string, collection_name, query, chain_type='stuff'):
    # Set up PGVector
    #embeddings = OpenAIEmbeddings()
    COLLECTION_NAME = collection_name
    CONNECTION_STRING = connection_string

    db = PGVector(
        connection_string=CONNECTION_STRING,
        embedding_function=OpenAIEmbeddings(),
        collection_name=COLLECTION_NAME,
        distance_strategy = DistanceStrategy.COSINE,
    )

    retriever = db.as_retriever(search_kwargs={"k": 1})

    llm = ChatOpenAI(temperature = 0.0, model = 'gpt-3.5-turbo-16k', max_tokens=13000,verbose=True)
    llm = ChatOpenAI(temperature = 0.0, model = 'gpt-4')

    qa_stuff = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type=chain_type, 
        retriever=retriever,
        verbose=True,
    )

    response = qa_stuff.invoke(query)

    print(response['result'])

def run_query(connection_string):
    # Establish a connection to the database
    with psycopg2.connect(connection_string) as conn:
        # Create a cursor object
        with conn.cursor() as cur:
            # Execute the SQL query
            cur.execute("""
                SELECT c.uuid, c.name, COUNT(e.*)
                FROM langchain_pg_collection c
                LEFT JOIN langchain_pg_embedding e ON c.uuid = e.collection_id
                GROUP BY c.uuid, c.name
                ORDER BY COUNT(e.*) DESC;
            """)
            # Fetch all the results
            results = cur.fetchall()
    # Return the results
    return results

def list_pgvector_collections(connection_string):
    try:
        # Run the SQL query and get the results
        collections = run_query(connection_string)
        
        # Display the results
        print("Listing PGVector Collections:")
        for uuid, name, size in collections:
            print(f"Collection UUID: {uuid} | Name: {name} | Size: {size}")
            
    except Exception as e:
        print(f"An error occurred while listing collections: {e}")


# Argument Parser
parser = argparse.ArgumentParser(description="RAGTool: Create, query and manage collections of documents using Retrieval-Augmented Generation via PGSQL pgvector and OpenAI. Authored by Buanzo.")
parser.add_argument('--openai-env', type=str, default='RAGTOOL_OPENAI_API_KEY', help='Environment variable for OpenAI API key.')
parser.add_argument('--pgenv', type=str, default='RAGTOOL_PGSQL_CONNECTION_STRING', help='Environment variable for PostgreSQL connection string.')
parser.add_argument('-v','--verbose', type=bool, default=False, help='Be more verbose.')
subparsers = parser.add_subparsers(dest='command')

# Create collection subcommand
create_parser = subparsers.add_parser('create', help='Create a new collection.')
create_parser.add_argument('-C', '--collection-name', required=True, type=str, help='Name of the collection to create.')
create_parser.add_argument('-s', '--source', required=True, type=str, help='Path to folder containing documents.')
create_parser.add_argument('--recreate', action='store_true', help='Flag to recreate the collection if it already exists.')

# Query collection subcommand
query_parser = subparsers.add_parser('query', help='Query an existing collection.')
query_parser.add_argument('-C', '--collection-name', required=True, type=str, help='Name of the collection to RAG query.')
query_parser.add_argument('--query', required=True, type=str, help='Query to run against the collection.')
query_parser.add_argument('--chain-type', default='stuff', choices=['stuff', 'refine', 'map_reduce', 'map_rerank'],
                          help='Type of chain method to use.')

# Delete collection subcommand
delete_parser = subparsers.add_parser('delete', help='Delete an existing collection.')
delete_parser.add_argument('-C', '--collection-name', required=True, type=str, help='Name of the collection to delete.')
delete_parser.add_argument('--force', action='store_true', help='Flag to forcefully delete the collection.')

# Check and Enable commands
needs_priv = f"Needs privileged PGSQL env var to be set. See --pgenv-privileged."
check_parser = subparsers.add_parser('pgvector-check', help=f'Check the PGSQL instance for pgvector extension. {needs_priv}')
enable_parser = subparsers.add_parser('pgvector-enable', help=f'Enable pgvector extension on PGSQL instance. {needs_priv}')

# List collection subcommand
list_parser = subparsers.add_parser('list', help='List existing collections.')

args = parser.parse_args()


# TODO: for debug mode, probably?
args_dict = vars(args)
formatted_args = json.dumps(args_dict, indent=4)
print(f"CLI Arguments:\n{formatted_args}")

connection_string = get_pgsql_connection_string(args.pgenv)

# Command Handlers
if args.command == 'create':
    create_pgvector_collection(connection_string, args.collection_name, args.source, recreate=args.recreate)

elif args.command == 'query':
    query_pgvector_collection(connection_string, args.collection_name, args.query, chain_type=args.chain_type)

elif args.command == 'delete':
    delete_pgvector_collection(connection_string, args.collection_name, force=args.force)

elif args.command == 'pgvector-check':
    check_pgvector_extension(connection_string)

elif args.command == 'pgvector-enable':
    enable_pgvector_extension(connection_string)

elif args.command == 'list':
    list_pgvector_collections(connection_string)

else:
    parser.print_help()
    sys.exit(1)
