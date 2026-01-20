# Agent Project

## Overview
The Agent project is a legal fact dense retrieval service built using FastAPI. It provides an efficient way to search through legal documents and retrieve relevant information based on user queries. The system utilizes advanced indexing techniques to ensure quick and accurate search results.

## Project Structure
```
agent
├── src
│   ├── main.py                # Entry point for the application
│   ├── config.py              # Configuration settings for the application
│   ├── indexing.py            # Indexing logic for the retrieval system
│   ├── logger.py              # Logging configuration
│   ├── test_dense_embedding.py # Test functions for search functionality
│   ├── api_client.py          # Client interface for interacting with the search API
│   └── utils
│       ├── __init__.py        # Init file for utils package
│       └── data_loader.py     # Utility functions for loading and processing data
├── tests
│   ├── test_search.py         # Unit tests for search functionality
│   └── test_api_client.py     # Unit tests for the API client
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
└── docs
    └── api_spec.md            # API specification documentation
```

## Installation
To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd agent
pip install -r requirements.txt
```

## Usage
To run the application, execute the following command:

```bash
python src/main.py
```

Once the server is running, you can access the API at `http://localhost:4241`.

## API Endpoints
The service exposes several endpoints for searching legal facts. Refer to `docs/api_spec.md` for detailed information on the available endpoints and their usage.

## Testing
To run the tests for the search functionality and API client, use the following command:

```bash
pytest tests/
```

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.