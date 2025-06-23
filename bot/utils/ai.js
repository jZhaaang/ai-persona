const { OpenAI } = require("openai");
const { Pinecone } = require("@pinecone-database/pinecone");
const axios = require("axios")

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY});

const index = pinecone.Index(process.env.PINECONE_INDEX);

async function getAIResponse(userPrompt) {
  const embedRes = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: userPrompt
  });
  const vector = embedRes.data[0].embedding;

  const queryRes = await index.query({
    topK: 5,
    vector,
    includeMetadata: true
  })

  const context = queryRes.matches
    .map((match) => `---\n${match.metadata.text}`)
    .join("\n\n")

  const chat = await openai.chat.completions.create({
    model: "gpt-4o",
    messages: [
      {
        role: "system",
        content: "You're simulating the personality of a Discord user named Abeyan (aliases: nayebaa, I can't read the sing) using their past messages. "
                + "You should respond to the user's prompt in the tone, grammar, and writing style of the character."
                + "This means not as much grammer, punctuation, formalities, and never any default emojis"
                + "Follow the style of his existing messages in the context below:"
      },
      {
        role: "user",
        content: `Context\n${context}\n\nUser: ${userPrompt}`
      }
    ]
  });

  return chat.choices[0].message.content.trim();
}

module.exports = { getAIResponse };