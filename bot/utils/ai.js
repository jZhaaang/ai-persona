const { OpenAI } = require("openai");
const { Pinecone } = require("@pinecone-database/pinecone");
const { encoding_for_model } = require("@dqbd/tiktoken");
const axios = require("axios");
const fs = require("fs");
const path = require("path");

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY});
const index = pinecone.Index(process.env.PINECONE_INDEX);
const encoder = encoding_for_model("gpt-4o");
const promptPath = path.join(__dirname, "../prompts/abeyan_prompt.txt");
const systemPrompt = fs.readFileSync(promptPath, "utf-8");

function countTokens(req) {
  return req.reduce((total, msg) => {
    const content = msg.content || "";
    return total + encoder.encode(content).length;
  }, 0);
}

async function getAIResponse(userPrompt) {
  const embedRes = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: userPrompt
  });
  const vector = embedRes.data[0].embedding;

  const queryRes = await index.query({
    topK: 5,
    vector,
    includeMetadata: true,
    filter: {
      author_names: { $in: ["Abeyan"] }
    }
  })

  const context = queryRes.matches
    .map((match) => `---\n${match.metadata.text}`)
    .join("\n\n")

  const messages = [
      {
        role: "system",
        content: systemPrompt
      },
      {
        role: "user",
        content: `Context\n${context}\n\nUser: ${userPrompt}`
      }
    ]

  const tokenCount = countTokens(messages);
  console.log(`${tokenCount} tokens\nUser: ${userPrompt}}`)
  console.log(`${context}\n\n`)

  const chat = await openai.chat.completions.create({
    model: "gpt-4o",
    messages: messages
  });

  return chat.choices[0].message.content.trim();
}

module.exports = { getAIResponse };