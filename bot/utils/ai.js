const { OpenAI } = require("openai");
const { Pinecone } = require("@pinecone-database/pinecone");
const { encoding_for_model } = require("@dqbd/tiktoken");
const axios = require("axios")

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY});
const index = pinecone.Index(process.env.PINECONE_INDEX);
const encoder = encoding_for_model("gpt-4o");

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
        content: "You're simulating the personality of a Discord user named Abeyan (aliases: nayebaa, I can't read the sing) using their past messages. "
                + "You should respond to the user's prompt in the tone, grammar, and writing style of the character."
                + "This means not as much grammer, punctuation, formalities, and never any emojis"
                + "Follow the style of his existing messages in the context below, try your best please!"
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