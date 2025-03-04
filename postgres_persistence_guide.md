# LangGraph PostgreSQL Persistence Guide

This guide explains how to set up and use PostgreSQL for persistent state storage with LangGraph, enabling your agents to maintain state across application restarts.

## Prerequisites

1. PostgreSQL database server installed and running
2. Python packages:
   - `psycopg` - PostgreSQL adapter for Python
   - `psycopg-pool` - Connection pooling for psycopg
   - `langgraph-checkpoint-postgres` - LangGraph PostgreSQL checkpointer

Install the required packages:

```bash
pip install psycopg psycopg-pool langgraph-checkpoint-postgres
```

## Database Setup

1. Create a PostgreSQL database for your LangGraph application:

```sql
CREATE DATABASE langgraph_db;
```

2. Create a user with appropriate permissions:

```sql
CREATE USER langgraph_user WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE langgraph_db TO langgraph_user;
```

3. Connection string format:

```
postgresql://username:password@hostname:port/database_name?sslmode=disable
```

Example:
```
postgresql://langgraph_user:your_secure_password@localhost:5432/langgraph_db?sslmode=disable
```

## Implementation Options

LangGraph offers multiple ways to connect to PostgreSQL for persistence:

### 1. Using a Connection String (Simplest)

```python
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://username:password@localhost:5432/langgraph_db?sslmode=disable"

# Create checkpointer
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    # Initialize tables (only needed first time)
    checkpointer.setup()
    
    # Create graph with checkpointer
    graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
    
    # Run with thread_id for persistence
    config = {"configurable": {"thread_id": "unique_conversation_id"}}
    result = graph.invoke(inputs, config)
```

### 2. Using a Connection Pool (Better for Production)

```python
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://username:password@localhost:5432/langgraph_db?sslmode=disable"
connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}

with ConnectionPool(
    conninfo=DB_URI,
    max_size=20,  # Adjust based on your application needs
    kwargs=connection_kwargs,
) as pool:
    checkpointer = PostgresSaver(pool)
    checkpointer.setup()  # Initialize tables
    
    graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "unique_conversation_id"}}
    result = graph.invoke(inputs, config)
```

### 3. Using a Direct Connection

```python
from psycopg import Connection
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://username:password@localhost:5432/langgraph_db?sslmode=disable"
connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}

with Connection.connect(DB_URI, **connection_kwargs) as conn:
    checkpointer = PostgresSaver(conn)
    checkpointer.setup()  # Initialize tables
    
    graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "unique_conversation_id"}}
    result = graph.invoke(inputs, config)
```

### 4. Async Implementation (For Async Applications)

```python
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

DB_URI = "postgresql://username:password@localhost:5432/langgraph_db?sslmode=disable"
connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}

async with AsyncConnectionPool(
    conninfo=DB_URI,
    max_size=20,
    kwargs=connection_kwargs,
) as pool:
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()  # Initialize tables
    
    graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "unique_conversation_id"}}
    result = await graph.ainvoke(inputs, config)
```

## Working with Checkpoints

### Retrieving Checkpoints

```python
# Get the latest checkpoint for a thread
checkpoint = checkpointer.get(config)

# List all checkpoints for a thread (for history)
checkpoint_tuples = list(checkpointer.list(config))
```

### Checkpoint Data Structure

Checkpoints contain:
- `id`: Unique identifier for the checkpoint
- `ts`: Timestamp when the checkpoint was created
- `channel_values`: The actual state data (messages, agent state, etc.)
- `versions_seen`: Version tracking for state changes
- `channel_versions`: Version tracking for channels

### Thread Management

Each conversation is identified by a `thread_id` in the `config`:

```python
config = {"configurable": {"thread_id": "unique_conversation_id"}}
```

You can:
- Create new threads with unique IDs
- Resume existing threads by using the same ID
- Track multiple conversations by using different IDs

## Production Considerations

1. **Connection Pooling**: Use connection pooling for better performance in production
2. **Security**: Store database credentials securely (environment variables, secrets manager)
3. **Backups**: Implement regular database backups
4. **Monitoring**: Monitor database performance and connection usage
5. **Scaling**: Consider read replicas for high-traffic applications

## Troubleshooting

Common issues:

1. **Connection Errors**: Check database credentials and network connectivity
2. **Permission Issues**: Ensure the database user has appropriate permissions
3. **Table Creation Failures**: Make sure `setup()` is called before first use
4. **Thread ID Conflicts**: Use unique thread IDs for different conversations
