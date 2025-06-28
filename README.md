# AI Persona
A Discord bot designed to emulate a specific user's personality and communication style using previous conversations. Using retrieval augmented generation, OpenAI's GPT models, and Pinecone for vector search, the bot pulls relevant historical messages from the user's chat history to generate personalized, context-aware responses.

## Features
* Personalized AI Conversations
  * Mimics a chosen user's tone and phrasing based on historical chat data
* Retrieval-Augmented Generation (RAG)
  * Uses Pinecone to search semantically similar past messages and provide accurate responses
  * Extracts and embeds prompt keywords in addition to the user's message to improve search relevance
* Contextual Memory
  * Maintains recent conversation turns for each user to allow for back-and-forth dialogue
* Discord Integration
  * Discord bot configured to respond for mentions or message replies


## Project Structure
```
ai-persona/
  bot/
    events/
      messageCreate.js
      ready.js
    prompts/
      {user}_prompt.txt
    utils/
      ai.js
  
  scripts/
    preprocess.py
    chunker.py
    embedder.py
    utils.py
    config.py

  data/
  .env
```

### Scripts
![image](https://github.com/user-attachments/assets/c059b794-6cdd-489d-bb77-93d26350abe6)
#### `preprocess.py`
Processes raw exported Discord messages and trims irrelevant attributes.
#### `chunker.py`
Formats processed message data into a readable messages and creates batched JSONL requests to feed to `gpt-4.1-mini`. Completed batches include chunks of conversations, represented as a list of `message_id` along with keywords for each conversation.
```
request = {
  "custom_id": f"batch_{i // CHUNK_BATCH_SIZE + 1}",
  "method": "POST",
  "url": "/v1/chat/completions",
  "body": {
      "model": CHUNK_MODEL,
      "messages": [
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": prompt},
      ],
  },
}
```
Here, `prompt` is a formatted array of messages in the form: `[{message_id}] ({timestamp}) {author_name}: {content}`. This provides the model with enough context to group messages based on time between messages, the people in the "conversation", and the content of the message itself. Our response contains only a list of `message_id` and `keywords` to minimize output tokens. A populated chunked file is then made using the message IDs containing a list of conversations.
#### `embedder.py`
The chunked conversations are embedded based on keywords and conversation messages. The conversation messages are, again, formatted in a readable way: `{author_name}: {content}`. With the embedding values, vector data is created consisting of the embeddings and keyword, authors, and messages metadata then uploaded to Pinecone.

### Bot
![image](https://github.com/user-attachments/assets/8eacf252-f861-4ff6-b636-3a41590e49b3)
#### `messageCreate.js`
The Discord bot consists of one `MessageCreate` event listener which listens for messages that **@mention** or **reply** to the bot. A map of message history between the user and bot is kept using the user's ID as a key. If a user is replying to the bot, it will poll this map and append the user's current message to the message history for more context on the current conversation. If there is no reply to the bot, the user's message is directly passed to the `ai.js` util to generate a response. Once a response is generated, the pair of messages between the user and the bot are saved to the map of message history for the user.
#### `ai.js`
Keywords are extracted based on the user's prompt, again using `gpt-4.1-mini`. The generated keywords and the user's original prompt are used to query the Pinecone database for the top 5 semantically similar conversation chunks (one query each for both the prompt and the keywords) for a total of 10 chunks. These chunks are then deduplicated and joined into one block of text to be included as context in the model's system prompt. Using a personality prompt, conversation history, and RAGed conversations, a response to the user's prompt is generated using the same model and returned to the user.

## Setup
### 1. Install Dependencies
```
# Python scripts for data processing
pip install -r requirements.txt

# Node.js for Discord bot
npm install
```
### 2. Configure `.env`
Your `.env` file should contain keys for OpenAI, Pinecone, and your Discord bot:
```
OPENAI_API_KEY=your-key
PINECONE_API_KEY=your-key
PINECONE_INDEX=your-index   # this can be any name
PINECONE_PROJECT_ID=your-id
DISCORD_TOKEN=your-token
```
### 3. Process Discord Messages
Add your exported Discord messages to a new folder, `data/raw/`, then run the scripts sequentially.
```
py scripts/preprocess.py
py scripts/chunker.py
py scrripts/embedder.py
```
> The batch API request process done in `chunker.py` is really inconsistent. Your batches will often fail with the error: "Enqueued token limit reached". More likely than not, this is inaccurate and you'll have no batches in queue, your queue limit just clears up very inconsistently. Haven't found a way around this other than retrying multiple times.
### 4. Run the Discord Bot
Assuming you have your bot setup and added to a server, you can start the bot in the `bot/` directory using `npm run dev` or just `node bot/index.js`.
## Future Ideas
* Create emulations of every user's personality and simulate a conversation entirely AI generated
