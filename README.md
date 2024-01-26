# RAGTool

RAGTool is a versatile Python-based utility designed for creating, querying, and managing collections of documents using Retrieval-Augmented Generation. It leverages PostgreSQL's pgvector extension and integrates seamlessly with OpenAI for advanced text processing and retrieval capabilities.

## Key Features

* *Collection Management:* Easily create, delete, and list collections of documents.
* *Document Retrieval:* Perform advanced queries on collections using retrieval-augmented generation techniques.
* *Extension Management:* Check and enable the PostgreSQL pgvector extension for enhanced functionality.
* *Flexible Configuration:* Configure OpenAI and PostgreSQL settings through environment variables for greater flexibility.


## Requirements

To get started with RAGTool, make sure you have the following prerequisites:

* Python 3
* PostgreSQL with the pgvector extension
* OpenAI API key

## Installation

To install the required Python packages, use the following requirements.txt file:

```
   langchain==0.1.1
   langchain_community==0.0.13
   langchain_openai==0.0.3
   psycopg2_binary==2.9.7
```
Install these packages using the following command:

```pip install -r requirements.txt```

## Usage

RAGTool offers a variety of functionalities that can be accessed through the command-line interface. Here are some common use cases:

### Create a Collection

Use the create subcommand to create a new collection:

```./ragtool.py create -C my_collection -s /path/to/documents --recreate```

- -C or --collection-name: Specify the name of the collection.
- -s or --source: Provide the path to the folder containing documents.
- --recreate: Use this flag to recreate the collection if it already exists.

### Query a Collection

Query an existing collection using the query subcommand:

```./ragtool.py query -C my_collection --query "Your query here" --chain-type stuff```

- -C or --collection-name: Specify the name of the collection to query.
- --query: Enter your query within double quotes.
- --chain-type: Choose the chain method (e.g., stuff, refine, map_reduce, map_rerank).

### Delete a Collection

Delete an existing collection using the delete subcommand:

```./ragtool.py delete -C my_collection --force```

- -C or --collection-name: Specify the name of the collection to delete.
- --force: Use this flag to forcefully delete the collection without confirmation.

### Check and Enable pgvector Extension

To check if the pgvector extension is installed and enable it if necessary, use the following subcommands:

#### Check the extension status:

```./ragtool.py pgvector-check```

#### Enable the extension:

```./ragtool.py pgvector-enable```

### List Existing Collections

#### List all existing collections:

```./ragtool.py list```

These are just some of the functionalities provided by RAGTool. Explore more options and customize your document management and retrieval tasks efficiently.

