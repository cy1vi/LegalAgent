# API Specification for Legal Fact Dense Retrieval Service

## Overview
This document outlines the API endpoints for the Legal Fact Dense Retrieval Service. The service provides functionality to search legal facts and retrieve relevant information based on user queries.

## Base URL
```
http://localhost:4241
```

## Endpoints

### 1. Search Endpoint

- **URL**: `/search`
- **Method**: `POST`
- **Description**: This endpoint allows users to search for legal facts based on a provided query.

#### Request Format
- **Content-Type**: `application/json`
- **Body**:
```json
{
  "fact": "string",          // The query string containing the legal fact to search for.
  "top_k": "integer"        // The number of top results to return (default is 10).
}
```

#### Response Format
- **Status Code**: `200 OK`
- **Body**:
```json
[
  {
    "fact_id": "string",     // Unique identifier for the fact.
    "score": "float",        // Relevance score of the result.
    "rank": "integer",       // Rank of the result in the returned list.
    "fact": "string",        // The text of the legal fact.
    "accusation": ["string"], // List of accusations related to the fact.
    "relevant_articles": ["string"], // List of relevant legal articles.
    "imprisonment": {        // Details about imprisonment terms, if applicable.
      "term": "string",       // Description of the imprisonment term.
      "details": "string"     // Additional details about the imprisonment.
    }
  }
]
```

#### Error Responses
- **Status Code**: `400 Bad Request`
  - **Body**:
  ```json
  {
    "detail": "string"       // Description of the error.
  }
  ```

- **Status Code**: `503 Service Unavailable`
  - **Body**:
  ```json
  {
    "detail": "Service initializing" // Indicates that the service is not ready.
  }
  ```

### 2. Batch Search Endpoint

- **URL**: `/batch_search`
- **Method**: `POST`
- **Description**: This endpoint allows users to perform batch searches for multiple legal facts in a single request.

#### Request Format
- **Content-Type**: `application/json`
- **Body**:
```json
{
  "facts": ["string"],       // An array of query strings containing legal facts to search for.
  "top_k": "integer"        // The number of top results to return for each query (default is 10).
}
```

#### Response Format
- **Status Code**: `200 OK`
- **Body**:
```json
[
  [
    {
      "fact_id": "string",   // Unique identifier for the fact.
      "score": "float",      // Relevance score of the result.
      "rank": "integer",     // Rank of the result in the returned list.
      "fact": "string",      // The text of the legal fact.
      "accusation": ["string"], // List of accusations related to the fact.
      "relevant_articles": ["string"], // List of relevant legal articles.
      "imprisonment": {      // Details about imprisonment terms, if applicable.
        "term": "string",     // Description of the imprisonment term.
        "details": "string"   // Additional details about the imprisonment.
      }
    }
  ]
]
```

#### Error Responses
- **Status Code**: `400 Bad Request`
  - **Body**:
  ```json
  {
    "detail": "string"       // Description of the error.
  }
  ```

- **Status Code**: `503 Service Unavailable`
  - **Body**:
  ```json
  {
    "detail": "Service initializing" // Indicates that the service is not ready.
  }
  ```

## Notes
- Ensure that the service is running before making requests to the API.
- The `top_k` parameter can be adjusted to retrieve more or fewer results based on user needs.
- The API is designed to handle both individual and batch search requests efficiently.