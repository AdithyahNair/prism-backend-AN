# AI Governance API Documentation

## Overview
This API provides endpoints for managing AI model governance, including auditing, benchmarking, and monitoring capabilities.

## Base URL
`http://localhost:8000`

## Authentication
All endpoints require authentication. Include the JWT token in the Authorization header:
```
Authorization: Bearer <token>
```

## Endpoints

### Projects
- `POST /projects/` - Create a new project
- `GET /projects/` - List all projects
- `GET /projects/{project_id}` - Get project details
- `PUT /projects/{project_id}` - Update project
- `DELETE /projects/{project_id}` - Delete project

### Models
- `POST /models/` - Create a new model entry
- `POST /models/{model_id}/upload` - Upload model file
- `GET /models/` - List all models
- `GET /models/{model_id}` - Get model details
- `PUT /models/{model_id}` - Update model
- `DELETE /models/{model_id}` - Delete model
- `POST /models/{model_id}/audit` - Run model audit
- `GET /models/{model_id}/audits` - List model audits
- `GET /models/{model_id}/audits/{audit_id}` - Get audit results

### Datasets
- `POST /datasets/` - Create a new dataset entry
- `POST /datasets/{dataset_id}/upload` - Upload dataset file
- `GET /datasets/` - List all datasets
- `GET /datasets/{dataset_id}` - Get dataset details
- `PUT /datasets/{dataset_id}` - Update dataset
- `DELETE /datasets/{dataset_id}` - Delete dataset
- `GET /datasets/{dataset_id}/stats` - Get dataset statistics
- `POST /datasets/{dataset_id}/validate` - Validate dataset

### Audits
- `POST /audits/` - Create a new audit
- `GET /audits/` - List all audits
- `GET /audits/{audit_id}` - Get audit details
- `PUT /audits/{audit_id}` - Update audit
- `DELETE /audits/{audit_id}` - Delete audit
- `GET /audits/{audit_id}/results` - Get audit results
- `GET /audits/{audit_id}/report` - Generate audit report

### Reports
- `POST /reports/` - Create a new report
- `GET /reports/` - List all reports
- `GET /reports/{report_id}` - Get report details
- `PUT /reports/{report_id}` - Update report
- `DELETE /reports/{report_id}` - Delete report
- `GET /reports/{report_id}/download` - Download report file

### Benchmarking
- `POST /benchmarks/` - Create a new benchmark
- `GET /benchmarks/` - List all benchmarks
- `GET /benchmarks/{benchmark_id}` - Get benchmark details
- `PUT /benchmarks/{benchmark_id}` - Update benchmark
- `DELETE /benchmarks/{benchmark_id}` - Delete benchmark
- `POST /benchmarks/{benchmark_id}/run` - Run benchmark
- `GET /benchmarks/{benchmark_id}/results` - Get benchmark results
- `GET /benchmarks/metrics` - List available metrics
- `GET /benchmarks/metrics/{metric_id}` - Get metric details

### LLM
- `POST /llm/connectors/` - Create a new LLM connector
- `GET /llm/connectors/` - List all connectors
- `GET /llm/connectors/{connector_id}` - Get connector details
- `PUT /llm/connectors/{connector_id}` - Update connector
- `DELETE /llm/connectors/{connector_id}` - Delete connector
- `POST /llm/evaluate` - Run LLM evaluation
- `GET /llm/evaluations` - List evaluations
- `GET /llm/evaluations/{evaluation_id}` - Get evaluation results

### Red Teaming
- `POST /red-team/tests/` - Create a new red team test
- `GET /red-team/tests/` - List all tests
- `GET /red-team/tests/{test_id}` - Get test details
- `PUT /red-team/tests/{test_id}` - Update test
- `DELETE /red-team/tests/{test_id}` - Delete test
- `POST /red-team/tests/{test_id}/run` - Run red team test
- `GET /red-team/tests/{test_id}/results` - Get test results

### Cookbooks
- `GET /cookbooks/` - List all cookbooks
- `GET /cookbooks/{cookbook_id}` - Get cookbook details
- `POST /cookbooks/{cookbook_id}/run` - Run cookbook
- `GET /cookbooks/{cookbook_id}/results` - Get cookbook results

## Error Handling
The API uses standard HTTP status codes:
- 200: Success
- 201: Created
- 400: Bad Request
- 401: Unauthorized
- 403: Forbidden
- 404: Not Found
- 500: Internal Server Error

Error responses include:
```json
{
    "detail": "Error message"
}
```

## Rate Limiting
API endpoints are rate-limited to:
- 100 requests per minute for authenticated users
- 10 requests per minute for unauthenticated users 