# LangGraph PostgreSQL Persistence

This project demonstrates how to implement persistent state storage for LangGraph agents using PostgreSQL, enabling human-in-the-loop conversations that can be resumed even after application restarts.

## Files in this Project

- `hitl_sample_2.py` - Original sample with in-memory persistence
- `postgres_hitl_example.py` - Complete example with PostgreSQL persistence
- `postgres_persistence_guide.md` - Comprehensive guide to PostgreSQL with LangGraph
- `docker-compose.yml` - Docker Compose file for setting up PostgreSQL
- `.env.example` - Template for environment variables

## Prerequisites

1. Python 3.9+
2. PostgreSQL database (can be set up using Docker Compose)
3. Required Python packages:
   ```
   pip install langgraph langchain-openai psycopg psycopg-pool langgraph-checkpoint-postgres python-dotenv
   ```

## Quick Start

### 1. Set up PostgreSQL

You can use the included Docker Compose file to set up PostgreSQL:

```bash
docker-compose up -d
```

This will create a PostgreSQL instance with:
- Database: `langgraph_db`
- Username: `langgraph_user`
- Password: `langgraph_password`
- Port: `5432`

### 2. Set up Environment Variables

Copy the example environment file and update it with your credentials:

```bash
cp .env.example .env
```

Edit the `.env` file to add your OpenAI API key.

### 3. Run the Example

```bash
python postgres_hitl_example.py
```

This will:
1. Connect to the PostgreSQL database
2. Set up the necessary tables
3. Allow you to create new conversations or resume existing ones
4. Persist all conversation state to the database

## Key Concepts

### Thread IDs

Each conversation is identified by a unique `thread_id`. To resume a conversation, you need to use the same thread ID.

### Checkpointing

LangGraph saves the entire state of the conversation graph at each step. This includes:
- All messages exchanged
- The current state of the agent
- Tool calls and their results
- Metadata about the conversation

### PostgreSQL Connection Methods

There are several ways to connect to PostgreSQL:

1. **Connection String** (simplest):
   ```python
   checkpointer = PostgresSaver.from_conn_string(DB_URI)
   ```

2. **Connection Pool** (recommended for production):
   ```python
   with ConnectionPool(conninfo=DB_URI, max_size=20) as pool:
       checkpointer = PostgresSaver(pool)
   ```

3. **Direct Connection**:
   ```python
   with Connection.connect(DB_URI) as conn:
       checkpointer = PostgresSaver(conn)
   ```

4. **Async Connection** (for async applications):
   ```python
   async with AsyncConnectionPool(conninfo=DB_URI) as pool:
       checkpointer = AsyncPostgresSaver(pool)
   ```

## Advanced Usage

For more advanced usage, including:
- Async implementations
- Connection pooling
- Checkpoint management
- Production considerations

See the `postgres_persistence_guide.md` file.

## Troubleshooting

### Common Issues

1. **Connection Errors**:
   - Check that PostgreSQL is running
   - Verify connection credentials
   - Ensure network connectivity

2. **Missing Tables**:
   - Make sure `checkpointer.setup()` is called before first use

3. **Thread ID Not Found**:
   - Verify you're using the correct thread ID
   - Check if the thread exists in the database

### Debugging

To debug connection issues, you can enable logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [psycopg Documentation](https://www.psycopg.org/psycopg3/docs/)
