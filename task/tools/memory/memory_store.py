import os
os.environ['OMP_NUM_THREADS'] = '1'

import json
from datetime import datetime, UTC, timedelta
import numpy as np
import faiss
from aidial_client import AsyncDial
from sentence_transformers import SentenceTransformer

from task.tools.memory._models import Memory, MemoryData, MemoryCollection


class LongTermMemoryStore:
    """
    Manages long-term memory storage for users.

    Storage format: Single JSON file per user in DIAL bucket
    - File: {user_id}/long-memories.json
    - Caching: In-memory cache with conversation_id as key
    - Deduplication: O(n log n) using FAISS batch search
    """

    DEDUP_INTERVAL_HOURS = 24

    def __init__(self, endpoint: str):
        #TODO:
        # 1. Set endpoint
        # 2. Create SentenceTransformer as model, model name is `all-MiniLM-L6-v2`
        # 3. Create cache, doct of str and MemoryCollection (it is imitation of cache, normally such cache should be set aside)
        # 4. Make `faiss.omp_set_num_threads(1)` (without this set up you won't be able to work in debug mode in `_deduplicate_fast` method
        # raise NotImplementedError()
        self.endpoint = endpoint
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self._cache: dict[str, MemoryCollection] = {}
        faiss.omp_set_num_threads(1)

    async def _get_memory_file_path(self, dial_client: AsyncDial) -> str:
        """Get the path to the memory file in DIAL bucket."""
        #TODO:
        # 1. Get DIAL app home path
        # 2. Return string with path in such format: `files/{bucket_with_app_home}/__long-memories/data.json`
        #    The memories will persist in appdata for this agent in `__long-memories` folder and `data.json` file
        #    (You will be able to check it also in Chat UI in attachments)
        # raise NotImplementedError()
        user_home = await dial_client.my_appdata_home()
        return f"files/{(user_home / '__long-memories/data.json').as_posix()}"

    async def _load_memories(self, api_key: str) -> MemoryCollection:
        #TODO:
        # 1. Create AsyncDial client (api_version is 2025-01-01-preview)
        # 2. Get memory file path
        # 3. Check cache: cache is dict of str and MemoryCollection, for the key we will use `memory file path` to make
        #    it simple. Such key will be unique for user and will allow to access memories across different
        #    conversations and only user can access them. In case if cache is present return its MemoryCollection.
        # ---
        # Below is logic when cache is not present:
        # 4. Open try-except block:
        #   - in try:
        #       - download file content
        #       - in response get content and decode it with 'utf-8'
        #       - load content with `json`
        #       - create MemoryCollection (it is pydentic model, use `model_validate` method)
        #   - in except:
        #       - create MemoryCollection (it will have empty memories, set up time for updated_at, more detailed take
        #         a look at MemoryCollection pydentic model and it Fields)
        # 5. Return created MemoryCollection
        # raise NotImplementedError()
        dial_client = AsyncDial(base_url=self.endpoint, api_key=api_key, api_version='2025-01-01-preview')
        file_path = await self._get_memory_file_path(dial_client)

        if file_path in self._cache:
            return self._cache[file_path]

        try:
            response = await dial_client.files.download(file_path)
            content = response.get_content().decode('utf-8')
            data = json.loads(content)
            collection = MemoryCollection.model_validate(data)
        except Exception as e:
            print(f"No existing memory file or error loading: {e}")
            collection = MemoryCollection()

        self._cache[file_path] = collection

        return collection

    async def _save_memories(self, api_key: str, memories: MemoryCollection):
        """Save memories to DIAL bucket and update cache."""
        #TODO:
        # 1. Create AsyncDial client
        # 2. Get memory file path
        # 3. Update `updated_at` of memories (now)
        # 4. Converts memories to json string (it's pydentic model and it have model dump json method for this). Don't
        #    make any indentations because it will make file 'bigger'. Here is the point that we store all the memories
        #    in one file and 'one memory' with its embeddings takes ~6-8Kb, we expect that there are won't be more that
        #    1000 memories but anyway for 1000 memories it will be ~6-8Mb, so, we need to make at least these small
        #    efforts to make it smaller 😉
        # 5. Put to cache (kind reminder the key is memory file path)
        # raise NotImplementedError()
        dial_client = AsyncDial(base_url=self.endpoint, api_key=api_key)
        file_path = await self._get_memory_file_path(dial_client)

        memories.updated_at = datetime.now(UTC)

        json_content = memories.model_dump_json()
        file_bytes = json_content.encode('utf-8')

        await dial_client.files.upload(url=file_path, file=file_bytes)

        self._cache[file_path] = memories

        print(f"Saved memories to {file_path}")

    async def add_memory(self, api_key: str, content: str, importance: float, category: str, topics: list[str]) -> str:
        """Add a new memory to storage."""
        #TODO:
        # 1. Load memories
        # 2. Make encodings for content with embedding model.
        #    Hint: provide content as list, and after encoding get first result (encode wil return list) and convertit `tolist`
        # 3. Create Memory
        #    - for id use `int(datetime.now(UTC).timestamp())` it will provide time now as int, it will be super enough
        #      to avoid collisions. Also, we won't use id but we added it because maybe in future you will make enhanced
        #      version of long-term memory and after that it will be additional 'headache' to add such ids 😬
        # 4. Add to memories created memory
        # 5. Save memories (it is PUT request bzw, -> https://dialx.ai/dial_api#tag/Files/operation/uploadFile)
        # 6. Return information that content has benn successfully stored
        # raise NotImplementedError()
        collection = await self._load_memories(api_key)

        embedding = self.model.encode([content])[0].tolist()

        memory = Memory(
            data=MemoryData(
                id=int(datetime.now(UTC).timestamp()),
                content=content,
                importance=importance,
                category=category,
                topics=topics
            ),
            embedding=embedding
        )

        collection.memories.append(memory)

        await self._save_memories(api_key, collection)

        return f"Successfully stored memory: {content}"

    async def search_memories(self, api_key: str, query: str, top_k: int = 5) -> list[MemoryData]:
        """
        Search memories using semantic similarity.

        Returns:
            List of MemoryData objects (without embeddings)
        """
        #TODO:
        # 1. Load memories
        # 2. If they are empty return empty array
        # ---
        # 3. Check if they needs_deduplication, if yes then deduplicate_and_save (need to implements both of these methods)
        # 4. Make vector search (embeddings are part of memory)😈
        # 5. Return `top_k` MemoryData based on vector search
        # raise NotImplementedError()
        collection = await self._load_memories(api_key)

        if not collection.memories:
            return []

        if self._needs_deduplication(collection):
            print("Deduplication needed, running now...")
            collection = await self._deduplicate_and_save(api_key, collection)

        embeddings = np.array([m.embedding for m in collection.memories]).astype('float32')
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / norms

        index = faiss.IndexFlatIP(normalized_embeddings.shape[1])
        index.add(normalized_embeddings)

        query_embedding = self.model.encode([query]).astype('float32')
        query_norm = np.linalg.norm(query_embedding, keepdims=True)
        normalized_query = query_embedding / query_norm

        k = min(top_k, len(collection.memories))
        similarities, indices = index.search(normalized_query, k)

        results = [collection.memories[i].data for i in indices[0]]

        return results

    def _needs_deduplication(self, collection: MemoryCollection) -> bool:
        """Check if deduplication is needed (>24 hours since last deduplication)."""
        #TODO:
        # The criteria for deduplication (collection length > 10 and >24 hours since last deduplication) or
        # (collection length > 10 last deduplication is None)
        # raise NotImplementedError()
        try:
            # If never deduplicated, trigger it
            if collection.last_deduplicated_at is None:
                return True

            time_since_dedup = datetime.now(UTC) - collection.last_deduplicated_at
            return time_since_dedup > timedelta(hours=self.DEDUP_INTERVAL_HOURS)
        except Exception as e:
            print(f"Error checking deduplication need: {e}")
            return False

    async def _deduplicate_and_save(self, api_key: str, collection: MemoryCollection) -> MemoryCollection:
        """
        Deduplicate memories synchronously and save the result.
        Returns the updated collection.
        """
        #TODO:
        # 1. Make fast deduplication (need to implement)
        # 2. Update last_deduplicated_at as now
        # 3. Save deduplicated memories
        # 4. Return deduplicated collection
        # raise NotImplementedError()
        try:
            original_count = len(collection.memories)

            if original_count < 2:
                return collection

            deduplicated_memories = self._deduplicate_fast(collection.memories)

            collection.memories = deduplicated_memories
            collection.last_deduplicated_at = datetime.now(UTC)

            await self._save_memories(api_key, collection)

            removed_count = original_count - len(deduplicated_memories)
            print(f"Deduplication complete: {original_count} -> {len(deduplicated_memories)} (removed {removed_count})")

            return collection

        except Exception as e:
            print(f"Error during deduplication: {e}")
            # Return original collection so search can continue
            return collection

    def _deduplicate_fast(self, memories: list[Memory]) -> list[Memory]:
        """
        Fast deduplication using FAISS batch search with cosine similarity.

        Strategy:
        - Find k nearest neighbors for each memory using cosine similarity
        - Mark duplicates based on similarity threshold (cosine similarity > 0.75)
        - Keep memory with higher importance
        """
        #TODO:
        # This is the hard part 🔥🔥🔥
        # You need to deduplicate memories, duplicates are the memories that have 75% similarity.
        # Among duplicates remember about `importance`, most important have more priorities to survive
        # It must be fast, it is possible to do for O(n log n), probably you can find faster way (share with community if do 😉)
        # Return deduplicated memories
        # raise NotImplementedError()
        if len(memories) < 2:
            return memories

        embeddings = np.array([m.embedding for m in memories]).astype('float32')
        n = len(embeddings)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / norms

        index = faiss.IndexFlatIP(normalized_embeddings.shape[1])
        index.add(normalized_embeddings)

        k = min(10, n)
        similarities, indices = index.search(normalized_embeddings, k)

        duplicates_to_remove = set()

        for i in range(n):
            if i in duplicates_to_remove:
                continue

            for j in range(1, k):
                neighbor_idx = indices[i][j]

                if neighbor_idx in duplicates_to_remove:
                    continue

                if similarities[i][j] > 0.75:
                    if memories[i].data.importance >= memories[neighbor_idx].data.importance:
                        duplicates_to_remove.add(neighbor_idx)
                    else:
                        duplicates_to_remove.add(i)
                        break

        deduplicated = [m for i, m in enumerate(memories) if i not in duplicates_to_remove]
        return deduplicated

    async def delete_all_memories(self, api_key: str, ) -> str:
        """
        Delete all memories for the user.

        Removes the memory file from DIAL bucket and clears the cache
        for the current conversation.
        """
        #TODO:
        # 1. Create AsyncDial client
        # 2. Get memory file path
        # 3. Delete file
        # 4. Return info about successful memory deletion
        raise NotImplementedError()
        try:
            dial_client = AsyncDial(base_url=self.endpoint, api_key=api_key)
            file_path = await self._get_memory_file_path(dial_client)

            try:
                await dial_client.files.delete(file_path)
                print(f"Deleted memory file: {file_path}")
            except Exception as e:
                print(f"Memory file not found or already deleted: {e}")

            if file_path in self._cache:
                del self._cache[file_path]
                print(f"Cleared memory cache: {file_path}")

            return "Successfully deleted all long-term memories."

        except Exception as e:
            error_msg = f"Error deleting memories: {e}"
            print(error_msg)
            return error_msg
