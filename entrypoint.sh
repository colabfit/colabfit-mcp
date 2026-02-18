#!/bin/bash
set -e

# if [ -n "$MONGODB_HOST" ]; then
#     echo "Waiting for MongoDB at ${MONGODB_HOST}:${MONGODB_PORT:-27017}..." >&2
#     MONGO_READY=0
#     for i in $(seq 1 30); do
#         if python3 -c "
# from pymongo import MongoClient
# c = MongoClient('${MONGODB_HOST}', ${MONGODB_PORT:-27017}, serverSelectionTimeoutMS=2000)
# c.admin.command('ping')
# " 2>/dev/null; then
#             echo "MongoDB is ready." >&2
#             MONGO_READY=1
#             break
#         fi
#         sleep 1
#     done
#     if [ "$MONGO_READY" -eq 0 ]; then
#         echo "WARNING: MongoDB not reachable after 30 attempts. Continuing without MongoDB." >&2
#     fi
# fi

exec colabfit-mcp
